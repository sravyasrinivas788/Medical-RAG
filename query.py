from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from groq import Groq
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from database import get_history,save_message
from datetime import datetime, timedelta


from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
 
load_dotenv()

embedder    = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker    = CrossEncoder("BAAI/bge-reranker-base")
qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),  
    port=int(os.getenv("QDRANT_PORT", 6333))
)
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
            "text":        p.payload.get("page_content") or p.payload.get("text", ""),
            "source":      p.payload.get("metadata", {}).get("source", ""),
            "source_type": p.payload.get("metadata", {}).get("source_type", ""),
            "heading":     p.payload.get("metadata", {}).get("heading", "")
        }
        for p in all_points
        if p.payload.get("page_content") or p.payload.get("text")
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
            "text":        r.payload.get("page_content") or r.payload.get("text", ""),
            "source":      r.payload.get("metadata", {}).get("source", ""),
            "source_type": r.payload.get("metadata", {}).get("source_type", ""),
            "score":       round(r.score, 3),
            "heading":     r.payload.get("metadata", {}).get("heading", "")
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


class MedicalResponse(BaseModel):
    answer: str = Field(description="Answer to the medical question")

parser = PydanticOutputParser(pydantic_object=MedicalResponse)

class HybridRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        chunks=search(query)
        docs=[]
        for c in chunks:
            docs.append(
                Document(
                    page_content=c["text"],
                    metadata={"source": c["source"], "source_type": c.get("source_type", ""), "heading": c.get("heading", "")}
                )
            )
        return docs


class GroqLLM(LLM):
    def _call(self, prompt: str, stop=None):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self):
        return "groq"




PROMPT=PromptTemplate(
    template="""You are a medical expert assistant. Use the following context to answer the question.
If you don't know the answer, just say "I don't know". Don't make up an answer.

Context:
{context}

Question: {question}

{format_instructions}

Answer:""",
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
REWRITE_PROMPT=PromptTemplate(
    template="""Rewrite the following question to be more specific and detailed based on the conversation history.

Conversation History:
{history}

Current Question: {question}

Rewritten Question:""",
    input_variables=["history", "question"]
)

reteriver=HybridRetriever()
llm=GroqLLM()

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata['source']}]\n{d.page_content}"
        for d in docs
    )



def rewrite_query(query:str,history):
    if not history:
        return query

    history_text="\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in history[-6:]
    )
    prompt=REWRITE_PROMPT.format(history=history_text,question=query)
    return llm.invoke(prompt)
    
    
def rewrite_steps(inputs):
    query=inputs["query"]
    session_id=inputs["session_id"]
    history=get_history(session_id)
    rewritten_query=rewrite_query(query,history)
    return{"query": rewritten_query, "session_id": session_id}



def retrive_step(inputs):
    query=inputs["query"]
    session_id=inputs["session_id"]
    docs=reteriver.invoke(query)
    context=format_docs(docs)
    return{"docs": docs, "session_id": session_id,"context":context,"history":inputs.get("history",[]),"query":query}

def save_memory(inputs,answer):
    save_message(inputs["session_id"], "user", inputs["query"])
    save_message(inputs["session_id"], "assistant", answer)
    



chain=(
    RunnableLambda(rewrite_steps) |
    RunnableLambda(retrive_step) |
    {
        "context": RunnableLambda(lambda x: x["context"]),
        "question": RunnableLambda(lambda x: x["query"])
    }
    | PROMPT
    | llm
    | parser
)

def ask(query: str, session_id:str):
    result=chain.invoke({"query": query, "session_id": session_id})
    save_memory({"session_id": session_id, "query": query}, result.answer)
    docs=reteriver.invoke(query)
    return{
        "answer": result.answer,
        "sources": [
            {
                "source":      d.metadata.get("source", ""),
                "source_type": d.metadata.get("source_type", ""),
                "heading":     d.metadata.get("heading", "")
            }
            for d in docs
        ]
    }














