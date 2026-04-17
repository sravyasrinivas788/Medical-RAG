from groq import Groq
from query import search, ask
from agent import run_agent, execute_tool
from database import get_conn, save_message, get_history
from psycopg2.extras import RealDictCursor
import os
import json
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API"))



AGENTS = {
    "clinical": {
        "name":        "Clinical agent",
        "description": "Handles questions about WHO guidelines, medicine categories, recommendations, essential medicines list, antibiotic groups, treatment protocols",
        "system":      """You are a clinical knowledge specialist with access to WHO Essential Medicines guidelines.
Answer questions using the medical knowledge base search tools.
Always cite the specific WHO guideline section you used.
If confidence is low, say so clearly — do not speculate on clinical matters."""
    },
    "drug": {
        "name":        "Drug agent",
        "description": "Handles questions about specific drug dosages, contraindications, side effects, drug interactions, prescribing information",
        "system":      """You are a pharmacology specialist with access to structured drug records.
Answer questions using the drug database tools.
Always provide exact dosages and contraindications from the database.
Always flag dangerous contraindications prominently.
If a drug is not in the database, say so explicitly."""
    },
    "admin": {
        "name":        "Admin agent",
        "description": "Handles questions about clinic policies, appointment procedures, prescription refills, referrals, lab test reference ranges",
        "system":      """You are a clinic administrative assistant with access to policies and lab ranges.
Answer questions about clinic procedures clearly and concisely.
For lab ranges, always include the normal range, units, and clinical notes.
Direct patients to consult a doctor for clinical interpretation."""
    }
}


def triage(query: str) -> str:
    agent_descriptions = "\n".join(
        f"- {key}: {info['description']}"
        for key, info in AGENTS.items()
    )

    prompt = f"""You are a medical query triage system.
Route the following query to the most appropriate specialist agent.

Available agents:
{agent_descriptions}

Query: {query}

Reply with ONLY one word — the agent key: clinical, drug, or admin.
If the query needs both clinical and drug knowledge, reply: drug
If unsure, reply: clinical"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )

    route = response.choices[0].message.content.strip().lower()
    if route not in AGENTS:
        route = "clinical"  # safe fallback

    print(f"[Triage] Query routed to: {route}")
    return route


def run_clinical_agent(query: str, session_id: str) -> dict:
    print("[Clinical Agent] Running...")
    try:
        # Use run_agent for iterative tool calling (same as /agent/ask endpoint)
        result = run_agent(query, session_id=f"{session_id}-clinical")
        
        # Determine confidence based on whether tools found useful information
        confidence = "high" if result.get("tools_used") else "medium"
        
        return {
            "agent":      "clinical",
            "answer":     result["answer"],
            "confidence": confidence,
            "sources":    result.get("tools_used", [])
        }
    except Exception as e:
        print(f"[Clinical Agent] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "agent":      "clinical",
            "answer":     f"I encountered an error while processing your request: {str(e)}",
            "confidence": "low",
            "sources":    []
        }

def run_drug_agent(query: str, session_id: str) -> dict:
    print("[Drug Agent] Running...")

    # extract drug name from query and do direct DB lookup first
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT name FROM drugs")
    all_drugs = [r["name"].lower() for r in cur.fetchall()]
    cur.close()
    conn.close()

    # find which drug is mentioned
    query_lower  = query.lower()
    matched_drug = next((d for d in all_drugs if d in query_lower), None)

    context_parts = []

    if matched_drug:
        # get exact DB record
        drug_result = execute_tool("get_drug_record", {"drug_name": matched_drug})
        context_parts.append(f"Drug record:\n{drug_result}")
        print(f"[Drug Agent] Found drug record: {matched_drug}")

    # also search for additional context
    search_results = search(query)
    if search_results:
        search_context = "\n\n".join(
            f"[{r['source']}]: {r['text']}"
            for r in search_results[:3]
        )
        context_parts.append(f"Additional context:\n{search_context}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant drug information found."

    messages = [
        {"role": "system", "content": AGENTS["drug"]["system"]},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1
    )

    top_score = search_results[0].get("rerank_score", 0) if search_results else 0
    confidence = "high" if top_score > 0.7 or matched_drug else "medium"

    return {
        "agent":      "drug",
        "answer":     response.choices[0].message.content,
        "confidence": confidence,
        "sources":    [{"source": f"Drug record: {matched_drug}"}] if matched_drug else []
    }

def run_admin_agent(query: str, session_id: str) -> dict:
    print("[Admin Agent] Running...")

    context_parts = []

    # check lab ranges
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT test_name FROM lab_ranges")
    lab_names   = [r["test_name"].lower() for r in cur.fetchall()]
    cur.execute("SELECT topic FROM clinic_policies")
    policy_topics = [r["topic"].lower() for r in cur.fetchall()]
    cur.close()
    conn.close()

    query_lower = query.lower()

    # check if lab test mentioned
    matched_lab = next((l for l in lab_names if l in query_lower), None)
    if matched_lab:
        lab_result = execute_tool("get_lab_range", {"test_name": matched_lab})
        context_parts.append(f"Lab range:\n{lab_result}")
        print(f"[Admin Agent] Found lab range: {matched_lab}")

    # check if policy topic mentioned
    matched_policy = next((p for p in policy_topics if p in query_lower), None)
    if matched_policy:
        policy_result = execute_tool("get_clinic_policy", {"topic": matched_policy})
        context_parts.append(f"Policy:\n{policy_result}")
        print(f"[Admin Agent] Found policy: {matched_policy}")

    # fallback to search
    if not context_parts:
        search_results = search(query)
        if search_results:
            context_parts.append("\n\n".join(r["text"] for r in search_results[:3]))

    context    = "\n\n".join(context_parts) if context_parts else "No relevant admin information found."
    confidence = "high" if matched_lab or matched_policy else "medium"

    messages = [
        {"role": "system", "content": AGENTS["admin"]["system"]},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1
    )

    return {
        "agent":      "admin",
        "answer":     response.choices[0].message.content,
        "confidence": confidence,
        "sources":    [{"source": f"Lab: {matched_lab or matched_policy or 'search'}"}]
    }


def needs_escalation(result: dict) -> bool:
    
    if result.get("confidence") == "low":
        return True
    uncertainty_phrases = [
        "i don't have enough information",
        "i cannot answer",
        "please consult",
        "not in the context"
    ]
    answer_lower = result.get("answer", "").lower()
    return any(phrase in answer_lower for phrase in uncertainty_phrases)

def escalate(query: str, result: dict, session_id: str) -> dict:
    print(f"[Escalation] Low confidence — flagging for human review")

    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            id          SERIAL PRIMARY KEY,
            session_id  TEXT,
            query       TEXT,
            agent       TEXT,
            answer      TEXT,
            created_at  TIMESTAMP DEFAULT NOW(),
            resolved    BOOLEAN DEFAULT FALSE
        )
    """)
    cur.execute("""
        INSERT INTO escalations (session_id, query, agent, answer)
        VALUES (%s, %s, %s, %s)
    """, (session_id, query, result.get("agent"), result.get("answer")))
    conn.commit()
    cur.close()
    conn.close()

    return {
        **result,
        "escalated": True,
        "escalation_message": "This query has been flagged for review by a medical professional. You will receive a response shortly."
    }


def run_multi_agent(query: str, session_id: str) -> dict:
    print(f"\n=== Multi-agent system starting ===")
    print(f"Query: '{query}'")

    # step 1 — triage
    route = triage(query)

    # step 2 — run specialist agent
    if route == "clinical":
        result = run_clinical_agent(query, session_id)
    elif route == "drug":
        result = run_drug_agent(query, session_id)
    elif route == "admin":
        result = run_admin_agent(query, session_id)
    else:
        result = run_clinical_agent(query, session_id)

    # step 3 — confidence check + escalation
    if needs_escalation(result):
        result = escalate(query, result, session_id)

    # step 4 — save to conversation history
    save_message(session_id, "user",      query)
    save_message(session_id, "assistant", result["answer"])

    return {
        "answer":     result["answer"],
        "agent_used": result["agent"],
        "confidence": result.get("confidence", "medium"),
        "escalated":  result.get("escalated", False),
        "sources":    result.get("sources", []),
        "session_id": session_id,
        "escalation_message": result.get("escalation_message")
    }