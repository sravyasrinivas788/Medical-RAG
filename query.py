from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from groq import Groq
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from database import save_message, get_history
from datetime import datetime, timedelta

load_dotenv()

embedder    = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker    = CrossEncoder("BAAI/bge-reranker-base")
qdrant      = QdrantClient(host="localhost", port=6333)
groq_client = Groq(api_key=os.getenv("GROQ_API"))
COLLECTION  = "medical_knowledge"

bm25_index  = None
bm25_corpus = []

def build_bm25_index():
    global bm25_index, bm25_corpus
    print("Building BM25 index from Qdrant...")
    all_points = []
    offset     = None

    while True:
        result, next_offset = qdrant.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        all_points.extend(result)
        if next_offset is None:
            break
        offset = next_offset

    bm25_corpus = [
        {
            "text":        p.payload["text"],
            "source":      p.payload["source"],
            "source_type": p.payload["source_type"],
            "heading":     p.payload.get("heading", "")
        }
        for p in all_points
    ]

    tokenized  = [doc["text"].lower().split() for doc in bm25_corpus]
    bm25_index = BM25Okapi(tokenized)
    print(f"BM25 index built — {len(bm25_corpus)} documents indexed.")


def dense_search(query: str, top_k: int = 10) -> list[dict]:
    vec     = embedder.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=top_k
    ).points
    return [
        {
            "text":        r.payload["text"],
            "source":      r.payload["source"],
            "source_type": r.payload["source_type"],
            "score":       round(r.score, 3),
            "heading":     r.payload.get("heading", "")
        }
        for r in results
    ]

def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    if bm25_index is None:
        build_bm25_index()

    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)

    scored = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [
        {
            "text":        bm25_corpus[i]["text"],
            "source":      bm25_corpus[i]["source"],
            "source_type": bm25_corpus[i]["source_type"],
            "score":       float(s),
            "heading":     bm25_corpus[i]["heading"]
        }
        for i, s in scored if s > 0
    ]

def rrf_merge(
    dense_results: list[dict],
    bm25_results:  list[dict],
    k: int = 60
) -> list[dict]:
    scores = {}
    docs   = {}

    for rank, doc in enumerate(dense_results):
        key = doc["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs[key]   = doc

    for rank, doc in enumerate(bm25_results):
        key = doc["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs[key]   = doc

    merged = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [
        {**docs[key], "rrf_score": round(scores[key], 4)}
        for key in merged
    ]

CONFIDENCE_THRESHOLDS = {
    "high":   0.7,
    "medium": 0.4,
    "low":    0.3,
}

def get_confidence_level(top_score: float) -> str:
    if top_score >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif top_score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"

def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    if not candidates:
        return []

    pairs  = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)

    ranked    = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    confident = [c for c in ranked if c["rerank_score"] >= CONFIDENCE_THRESHOLDS["low"]]

    # always return at least 1 chunk even if all below threshold
    return confident[:top_n] if confident else ranked[:1]

def clean_history_message(content: str) -> str:
    # remove the Context block from stored messages
    # history messages were saved as "Context:\n...\n\nQuestion: X"
    # we only want "Question: X" for history
    if "Question:" in content and "Context:" in content:
        # extract just the question part
        parts = content.split("Question:")
        if len(parts) > 1:
            return "Question:" + parts[-1].strip()
    return content

def search(query: str) -> list[dict]:
    dense  = dense_search(query, top_k=10)
    bm25   = bm25_search(query,  top_k=10)
    merged = rrf_merge(dense, bm25)
    final  = rerank(query, merged[:10], top_n=5)
    return final

def rewrite_query_history(query: str, history: list[dict]) -> str:
    if not history:
        return query

    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in history[-10:]
    )

    prompt = f"""Given this conversation history and a follow-up question,
rewrite the follow-up question to be self-contained and clear.
If the follow-up already makes sense standalone, return it unchanged.
Do NOT add speculation or extra context that wasn't in the original question.
Preserve the exact terms and spelling from the original question.
Return ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question: {query}
Rewritten question:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def ask(query: str, session_id: str) -> dict:
    # get history and check if it's recent (within 8 hours)
    raw_history = get_history(session_id)
    
    # Filter history by time - only use if last message is within 8 hours
    history = []
    if raw_history:
        last_message_time = raw_history[-1].get("created_at")
        if last_message_time:
            # Handle both timezone-aware and naive datetimes
            now = datetime.now(last_message_time.tzinfo) if last_message_time.tzinfo else datetime.now()
            time_diff = now - last_message_time
            
            # Only use history if last message is within 8 hours
            if time_diff <= timedelta(hours=8):
                history = raw_history
                print(f"Using conversation history (last message: {time_diff.total_seconds()/3600:.1f} hours ago)")
            else:
                print(f"Ignoring old conversation history (last message: {time_diff.total_seconds()/3600:.1f} hours ago)")
    
    # Only rewrite if there are at least 2 previous exchanges (4+ messages)
    if len(history) >= 4:
        rewritten_query = rewrite_query_history(query, history)
        if rewritten_query != query:
            print(f"Query rewritten: '{query}' → '{rewritten_query}'")
    else:
        rewritten_query = query

    # search with rewritten query
    chunks = search(rewritten_query)

    if not chunks:
        answer = "I don't have enough information to answer this."
        save_message(session_id, "user",      query)
        save_message(session_id, "assistant", answer)
        return {
            "answer":           answer,
            "sources":          [],
            "confidence":       "low",
            "session_id":       session_id,
            "rewritten_query":  rewritten_query if rewritten_query != query else None
        }

    top_score  = chunks[0]["rerank_score"]
    confidence = get_confidence_level(top_score)

    # Use all chunks returned by rerank
    strong_chunks = chunks
    
    context = "\n\n".join(
        f"[{i+1}] Source: {c['source']} (type: {c['source_type']}) | relevance: {c['rerank_score']}\n{c['text']}"
        for i, c in enumerate(strong_chunks)
    )

    print(f"\n=== DEBUG: Context being sent to LLM ===")
    print(f"Query: {query}")
    print(f"Confidence: {confidence} (top score: {top_score})")
    print(f"Context:\n{context}\n")

    system_msg = """You are a medical knowledge assistant for a clinic.
Answer questions using only the context provided.
If the answer is not in the context, say exactly: "I don't have enough information to answer this."
Do not speculate. Do not explain symbols or codes you don't understand.
Always mention which source(s) you used at the end of your answer.
Format answers clearly with numbered points where applicable."""

    messages = [{"role": "system", "content": system_msg}]

    for h in history[-10:]:
        cleaned_content = clean_history_message(h["content"])
        messages.append({"role": h["role"], "content": cleaned_content})

    
    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    if confidence == "medium":
        user_content += "\n\nNote: Available information is limited. Prefix your answer with 'Based on limited available information:'"
    
    # Add note if query was rewritten to help LLM understand context
    if rewritten_query != query:
        user_content += f"\n\n(Note: The user's question may contain typos or informal terms. Interpret the question based on the context provided.)"

    messages.append({"role": "user", "content": user_content})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    save_message(session_id, "user",      query)
    save_message(session_id, "assistant", answer)

    return {
        "answer":          answer,
        "confidence":      confidence,
        "session_id":      session_id,
        "rewritten_query": rewritten_query if rewritten_query != query else None,
        "sources": [
            {
                "source":       c["source"],
                "source_type":  c["source_type"],
                "rerank_score": c.get("rerank_score", 0),
                "section":      c.get("heading", ""),
                "page":         c.get("page", "")
            }
            for c in strong_chunks
        ]
    }