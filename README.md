# Medical Knowledge Base RAG System

Production-grade RAG system built on WHO Essential Medicines data.

## Stack
- FastAPI + PostgreSQL + Qdrant
- BGE embeddings + BM25 hybrid search
- Cross-encoder reranker
- Agentic RAG with tool calling
- MCP server
- Multi-agent triage system

## Run locally

1. Clone the repo
2. Copy .env.example to .env.production and fill in values
3. docker-compose up -d
4. Upload WHO PDF: curl -X POST http://localhost:8000/upload -F "file=@Medicines-List.pdf"
5. Ask questions: POST http://localhost:8000/ask

## Endpoints
- POST /ask — RAG Q&A
- POST /agent/ask — Agentic RAG
- POST /multi-agent/ask — Multi-agent triage
- GET /mcp/sse — MCP server