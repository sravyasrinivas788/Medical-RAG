from groq import Groq
from query import search, dense_search
from database import get_conn, save_message, get_history
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API"))
MAX_ITERATIONS = 5  

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": """Search the medical knowledge base using hybrid search.
Searches both the WHO Essential Medicines PDF and structured database records
(drugs, dosages, contraindications, lab ranges, clinic policies).
Use this when you need to find information to answer the user's question.
You can call this multiple times with different queries if needed.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific — include drug names, medical terms, condition names."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you are searching for this — helps track agent reasoning."
                    }
                },
                "required": ["query", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_drug_record",
            "description": """Get the complete structured record for a specific drug from the database.
Returns name, category, indication, dosage, contraindications, and side effects.
Use this when you need precise dosage or contraindication data for a specific drug.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "The exact name of the drug to look up."
                    }
                },
                "required": ["drug_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_range",
            "description": """Get the normal reference range for a specific lab test.
Use this when the user asks about normal values, lab interpretation, or test thresholds.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "description": "The name of the lab test e.g. HbA1c, creatinine, INR."
                    }
                },
                "required": ["test_name"]
            }
        }
    }
]


def execute_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "search_knowledge_base":
        query   = tool_args["query"]
        reason  = tool_args.get("reason", "")
        print(f"  [Tool] search_knowledge_base: '{query}' | reason: {reason}")

        results = search(query)
        if not results:
            return "No relevant results found for this query."

        # format results for the agent to read
        formatted = "\n\n".join(
            f"Result {i+1} (score: {r.get('rerank_score', 0)}, "
            f"source: {r['source']}, section: {r.get('heading', 'N/A')}):\n{r['text']}"
            for i, r in enumerate(results)
        )
        return formatted

    elif tool_name == "get_drug_record":
        drug_name = tool_args["drug_name"]
        print(f"  [Tool] get_drug_record: '{drug_name}'")

        conn = get_conn()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM drugs WHERE LOWER(name) LIKE LOWER(%s)",
            (f"%{drug_name}%",)
        )
        drug = cur.fetchone()
        cur.close()
        conn.close()

        if not drug:
            return f"No drug record found for '{drug_name}'."

        return (
            f"Drug: {drug['name']}\n"
            f"Category: {drug['category']}\n"
            f"Indication: {drug['indication']}\n"
            f"Dosage: {drug['dosage']}\n"
            f"Contraindications: {drug['contraindications']}\n"
            f"Side effects: {drug['side_effects']}"
        )

    elif tool_name == "get_lab_range":
        test_name = tool_args["test_name"]
        print(f"  [Tool] get_lab_range: '{test_name}'")

        conn = get_conn()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM lab_ranges WHERE LOWER(test_name) LIKE LOWER(%s)",
            (f"%{test_name}%",)
        )
        lab = cur.fetchone()
        cur.close()
        conn.close()

        if not lab:
            return f"No lab range found for '{test_name}'."

        return (
            f"Test: {lab['test_name']}\n"
            f"Normal range: {lab['normal_range']} {lab['unit']}\n"
            f"Notes: {lab['notes']}"
        )

    return f"Unknown tool: {tool_name}"


def run_agent(query: str, session_id: str) -> dict:
    history  = get_history(session_id)
    iteration = 0
    tool_calls_made = []

    print(f"\n=== Agent starting for query: '{query}' ===")

    # build initial messages
    system_msg = """You are a medical knowledge assistant for a clinic.
You have access to tools to search a medical knowledge base.

CRITICAL RULES:
- ONLY use information returned by the tools. Do NOT use your training data or general medical knowledge.
- If ALL tools return "No results found", say "I don't have enough information."
- If at least one tool returns useful information, answer based on that information.
- Always search before answering — never answer from memory alone.
- If your first search doesn't give enough information, search again with a different query.
- Once you have sufficient information from the tools, provide a clear and cited answer.
- Always mention which sources you used."""

    messages = [{"role": "system", "content": system_msg}]

    # add conversation history
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})

    # add current query
    messages.append({"role": "user", "content": query})

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Force tool use on first iteration to ensure agent searches before answering
        # tool_choice_setting = "required" if iteration == 1 else "auto"
        
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2
            )
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            answer = "I encountered an error while processing your request. Please try again."
            save_message(session_id, "user", query)
            save_message(session_id, "assistant", answer)
            return {
                "answer": answer,
                "session_id": session_id,
                "iterations": iteration,
                "tools_used": tool_calls_made,
                "agent_mode": True,
                "error": str(e)
            }

        msg = response.choices[0].message

        # check if agent wants to call tools
        if msg.tool_calls:
            # add agent's tool call decision to messages
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            # execute each tool call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                result = execute_tool(tool_name, tool_args)

                tool_calls_made.append({
                    "tool":   tool_name,
                    "args":   tool_args,
                    "result": result[:200]  # truncate for response
                })

                # add tool result back to messages
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result
                })

        else:
            # no tool calls — agent has produced final answer
            print(f"Agent finished in {iteration} iterations")
            answer = msg.content

            # save to history
            save_message(session_id, "user",      query)
            save_message(session_id, "assistant", answer)

            return {
                "answer":      answer,
                "session_id":  session_id,
                "iterations":  iteration,
                "tools_used":  tool_calls_made,
                "agent_mode":  True
            }

    # max iterations reached — return what we have
    print(f"Max iterations ({MAX_ITERATIONS}) reached")
    answer = "I was unable to find sufficient information to answer this question confidently."
    save_message(session_id, "user",      query)
    save_message(session_id, "assistant", answer)

    return {
        "answer":     answer,
        "session_id": session_id,
        "iterations": iteration,
        "tools_used": tool_calls_made,
        "agent_mode": True
    }