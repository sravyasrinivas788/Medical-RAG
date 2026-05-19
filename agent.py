from groq import Groq
from query import search, dense_search
from database import get_conn, save_message, get_history
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv
from typing import TypedDict,List,Dict,Any
from langgraph.graph import StateGraph, END
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
            return "No relevant results found for this query.", []

        formatted = "\n\n".join(
            f"Result {i+1} (score: {r.get('rerank_score', 0)}, "
            f"source: {r['source']}, section: {r.get('heading', 'N/A')}):\n{r['text']}"
            for i, r in enumerate(results)
        )
        sources = [
            {"source": r.get("source",""), "source_type": r.get("source_type",""), "heading": r.get("heading","")}
            for r in results
        ]
        return formatted, sources

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
            return f"No drug record found for '{drug_name}'.", []

        return (
            f"Drug: {drug['name']}\n"
            f"Category: {drug['category']}\n"
            f"Indication: {drug['indication']}\n"
            f"Dosage: {drug['dosage']}\n"
            f"Contraindications: {drug['contraindications']}\n"
            f"Side effects: {drug['side_effects']}"
        ), [{"source": f"db:drugs:{drug['name']}", "source_type": "db_drug", "heading": ""}]

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
            return f"No lab range found for '{test_name}'.", []

        return (
            f"Test: {lab['test_name']}\n"
            f"Normal range: {lab['normal_range']} {lab['unit']}\n"
            f"Notes: {lab['notes']}"
        ), [{"source": f"db:lab_ranges:{lab['test_name']}", "source_type": "db_lab", "heading": ""}]

    return f"Unknown tool: {tool_name}", []

class AgentState(TypedDict):
    query: str
    session_id: str
    messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    iteration: int
    last_message: Any
    sources: List[Dict[str, Any]]
    answer: str


def init_state(query:str,session_id:str):
    history=get_history(session_id)
    system_msg="""You are a medical knowledge assistant for a clinic.
You have access to tools to search a medical knowledge base.

CRITICAL RULES:
- ONLY use information returned by the tools. Do NOT use your training data or general medical knowledge.
- If ALL tools return "No results found", say "I don't have enough information."
- If at least one tool returns useful information, answer based on that information.
- Always search before answering — never answer from memory alone.
- If your first search doesn't give enough information, search again with a different query.
- Once you have sufficient information from the tools, provide a clear and cited answer.
- Always mention which sources you used."""
    messages=[{"role":"system","content":system_msg}]
    for h in history[-6:]:
        messages.append({"role":h["role"],"content":h["content"]})
    messages.append({"role":"user","content":query})
    return{
        "query":query,
        "session_id":session_id,
        "messages":messages,
        "tool_calls":[],
        "iteration":0,
        "last_message":None,
        "sources":[],
        "answer":""
    }

def llm_node(state:AgentState):
    state["iteration"] += 1
    response=groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=state["messages"],
        tools=TOOLS,
        tool_choice="auto"
    )
    answer=response.choices[0].message
    state["last_message"]=answer
    state["messages"].append({
        "role":"assistant",
        "content":answer.content,
        "tool_calls":answer.tool_calls if answer.tool_calls else None
    })
    return state

def tool_node(state:AgentState):
    answer=state["last_message"]
    if answer.tool_calls:
        for tool_call in answer.tool_calls:
            tool_name=tool_call.function.name
            tool_args=json.loads(tool_call.function.arguments)
            result, sources=execute_tool(tool_name,tool_args)

            state["tool_calls"].append({
                "tool_name":tool_name,
                "tool_args":tool_args,
                "result":result
            })

            for src in sources:
                if src not in state["sources"]:
                    state["sources"].append(src)

            state["messages"].append({
                "role":"tool",
                "content":result,
                "tool_call_id":tool_call.id
            })
    return state

def should_continue(state:AgentState):
    msg=state["last_message"]
    if not msg.tool_calls or state["iteration"] >= MAX_ITERATIONS:
        return "end"
    else:
        return "tool_node"

def final_node(state:AgentState):
    msg=state["last_message"]
    answer=msg.content or "No response"
    save_message(state["session_id"],"user",state["query"])
    save_message(state["session_id"],"assistant",answer)
    state["answer"]=answer
    return state

    
           
builder=StateGraph(AgentState)

builder.add_node("llm",llm_node)
builder.add_node("tools",tool_node)
builder.add_node("final",final_node)

builder.set_entry_point("llm")
builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tool_node":"tools",
        "end":"final"
    }
)
builder.add_edge("tools","llm")
builder.add_edge("final",END)
graph=builder.compile()

def run_agent(query:str,session_id: str):
    state=init_state(query,session_id)
    result=graph.invoke(state)
    return {
        "answer":     result.get("answer","No response"),
        "tools_used": [tc["tool_name"] for tc in result.get("tool_calls",[])],
        "sources":    result.get("sources",[]),
        "session_id": result.get("session_id",session_id)
    }
