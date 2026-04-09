from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from database import setup_tables, seed_dummy_data, save_file, get_all_files
from ingest import ingest_all, ingest_pdfs, setup_qdrant
from query import ask,build_bm25_index

app = FastAPI(title="Medical Knowledge Base")

ALLOWED_TYPES = {"pdf", "txt"}

@app.on_event("startup")
def startup():
    # setup_tables()
    # seed_dummy_data()
    # setup_qdrant()
    # ingest_all()  
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

@app.post("/ask")
async def ask_question(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    return ask(req.query)

@app.get("/documents")
async def list_documents():
    files = get_all_files()
    return [
        {"id": f["id"], "name": f["name"], "type": f["file_type"]}
        for f in files
    ]

@app.get("/health")
async def health():
    return {"status": "ok"}