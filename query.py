from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
qdrant     = QdrantClient(host="localhost", port=6333)
groq_client = Groq(api_key=os.getenv("GROQ_API"))
COLLECTION = "medical_knowledge"

def retrival(query: str, top_k: int = 5) -> list[dict]:
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
def ask(query: str) -> dict:
    chunks  = retrival(query)

   
    context = "\n\n".join(
        f"[{i+1}] Source: {c['source']} (type: {c['source_type']}) | relevance: {c['score']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""You are a medical knowledge assistant for a clinic.
Answer the question using only the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."
Always mention which source(s) you used at the end of your answer.

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
                "score":       c["score"]
            }
            for c in chunks
        ]
    }