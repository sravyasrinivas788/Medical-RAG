from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from groq import Groq
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
embedder   = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker   = CrossEncoder("BAAI/bge-reranker-base")
qdrant     = QdrantClient(host="localhost", port=6333)
groq_client = Groq(api_key=os.getenv("GROQ_API"))
COLLECTION = "medical_knowledge"

bm25_index   = None
bm25_corpus  = []  

def build_bm25_index():
    global bm25_index, bm25_corpus

    print("Building BM25 index from Qdrant...")
    all_points = []
    offset = None

    
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
            "source_type": p.payload["source_type"]
        }
        for p in all_points
    ]

    # tokenize for BM25 — lowercase, split on whitespace
    tokenized = [doc["text"].lower().split() for doc in bm25_corpus]
    bm25_index = BM25Okapi(tokenized)
    print(f"BM25 index built — {len(bm25_corpus)} documents indexed.")


def dense_search(query: str, top_k: int = 5) -> list[dict]:
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
            "score":       round(r.score, 3)
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
            "score":       float(s)
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
RERANK_THRESHOLD = 0.3
def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    if not candidates:
        return []

    pairs  = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
    # ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    # confident = [c for c in ranked if c["rerank_score"] >= RERANK_THRESHOLD]

    # if not confident:
    #     return ranked[:1]

    # return confident[:top_n]
def search(query: str) -> list[dict]:
    
    dense   = dense_search(query, top_k=10)
    bm25    = bm25_search(query,  top_k=10)

    merged  = rrf_merge(dense, bm25)

    final   = rerank(query, merged[:10], top_n=5)

    return final
def ask(query: str) -> dict:
    chunks  = search(query)

   
    context = "\n\n".join(
        f"[{i+1}] Source: {c['source']} (type: {c['source_type']}) | relevance: {c['score']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""You are a medical knowledge assistant for a clinic.
Answer the question using only the context provided below.
If the answer is not in the context, say exactly: "I don't have enough information to answer this."
Do not speculate. Do not explain symbols or codes you don't understand.
Always mention which source(s) you used at the end of your answer.
Format your answer clearly with numbered points where applicable.

Context:
{context}

Question: {query}
Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "source":      c["source"],
                "source_type": c["source_type"],
                "score":       c.get("rerank_score",0)
            }
            for c in chunks
        ]
    }
