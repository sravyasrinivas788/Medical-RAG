import fitz
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance,PointStruct
from database import get_all_drugs,get_all_lab_ranges,get_all_files,get_all_policies

embedder= SentenceTransformer('all-MiniLM-L6-v2')
qdrant=QdrantClient(url="http://localhost:6333")
COLLECTION_NAME="medical_knowledge"

def setup_qdrant():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("Qdrant collection created.")

        
def extract_pdf_text(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size=150, overlap=20) -> list[str]:
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]  

def ingest_pdfs():
    files = get_all_files()
    if not files:
        print("No PDFs found in documents table, skipping.")
        return

    for f in files:
        name    = f["name"]
        content = bytes(f["content"])
        text    = extract_pdf_text(content)
        chunks  = chunk_text(text)

        if not chunks:
            print(f"No text extracted from {name}, skipping.")
            continue

        embeddings = embedder.encode(chunks, show_progress_bar=True)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "text":        chunk,
                    "source":      name,
                    "source_type": "pdf",
                    "doc_id":      f["id"]
                }
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Ingested {len(points)} chunks from PDF: {name}")

def ingest_policies():
    policies = get_all_policies()
    if not policies:
        print("No policies found.")
        return

    points = []
    for p in policies:
        text = f"Clinic policy — {p['topic']}: {p['description']}"
        embedding = embedder.encode(text).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text":        text,
                "source":      f"Clinic policy: {p['topic']}",
                "source_type": "db_policy",
                "policy_id":   p["id"]
            }
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} clinic policies.") 

def ingest_drugs():
    drugs = get_all_drugs()
    if not drugs:
        print("No drugs found.")
        return

    points = []
    for d in drugs:
        text = (
            f"Drug: {d['name']}. "
            f"Category: {d['category']}. "
            f"Used for: {d['indication']}. "
            f"Dosage: {d['dosage']}. "
            f"Contraindications: {d['contraindications']}. "
            f"Side effects: {d['side_effects']}."
        )
        embedding = embedder.encode(text).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text":        text,
                "source":      f"Drug record: {d['name']}",
                "source_type": "db_drug",
                "drug_id":     d["id"]
            }
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} drug records.")

def ingest_lab_ranges():
    labs = get_all_lab_ranges()
    if not labs:
        print("No lab ranges found.")
        return

    points = []
    for l in labs:
        text = (
            f"Lab test: {l['test_name']}. "
            f"Normal range: {l['normal_range']} {l['unit']}. "
            f"Clinical notes: {l['notes']}."
        )
        embedding = embedder.encode(text).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text":        text,
                "source":      f"Lab reference: {l['test_name']}",
                "source_type": "db_lab",
                "lab_id":      l["id"]
            }
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} lab reference ranges.")

def ingest_all():
    setup_qdrant()
    print("\n--- Ingesting DB rows ---")
    ingest_drugs()
    ingest_policies()
    ingest_lab_ranges()
    print("\n--- Ingesting PDFs ---")
    ingest_pdfs()
    print("\nAll done.")

if __name__ == "__main__":
    ingest_all()
