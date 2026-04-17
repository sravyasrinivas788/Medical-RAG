from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from database import setup_tables, seed_dummy_data, save_file, get_all_files,clear_history
from ingest import ingest_all, ingest_pdfs, setup_qdrant
from query import ask,build_bm25_index
from agent import run_agent
from mcp_server import mcp_app
from multi_agent import run_multi_agent
app = FastAPI(title="Medical Knowledge Base")
app.mount("/mcp", mcp_app)

ALLOWED_TYPES = {"pdf", "txt"}

@app.on_event("startup")
def startup():
    setup_tables()
    seed_dummy_data()
    setup_qdrant()
    ingest_all()  
    build_bm25_index()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported type: .{ext}. Use PDF or TXT.")

    content = await file.read()
    doc_id  = save_file(file.filename, ext, content)

    
    ingest_pdfs()
    build_bm25_index()

    return {
        "status": "uploaded and ingested",
        "doc_id": doc_id,
        "file":   file.filename
    }

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
#initial RAG endpoint
@app.post("/ask")
async def ask_question(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    session_id=req.session_id
    result=ask(req.query, session_id)
    return result

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    from database import get_history
    history = get_history(session_id, last_n=20)
    return {"session_id": session_id, "history": history}
    
@app.get("/documents")
async def list_documents():
    files = get_all_files()
    return [
        {"id": f["id"], "name": f["name"], "type": f["file_type"]}
        for f in files
    ]

#Single-agent RAG
@app.post("/agent/ask")
async def agent_ask(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    session_id=req.session_id
    result=run_agent(req.query, session_id)
    return result
    

@app.get("/health")
async def health():
    return {"status": "ok"}


class MultiAgentRequest(BaseModel):
    query:      str
    session_id: str = None


#Multi-agent RAG
@app.post("/multi-agent/ask")
async def multi_agent_ask(req: MultiAgentRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    session_id = req.session_id or str(uuid.uuid4())
    return run_multi_agent(req.query, session_id=session_id)

@app.get("/escalations")
async def get_escalations():
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM escalations ORDER BY created_at DESC LIMIT 20")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"escalations": [dict(r) for r in rows]}

@app.patch("/escalations/{escalation_id}/resolve")
async def resolve_escalation(escalation_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE escalations SET resolved = TRUE WHERE id = %s", (escalation_id,))
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "resolved", "id": escalation_id}