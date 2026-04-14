import fitz
import uuid
import re
import io
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from database import get_all_drugs, get_all_lab_ranges, get_all_files, get_all_policies
import pdfplumber
embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "medical_knowledge"



def setup_qdrant():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print("Qdrant collection created.")

def is_heading(line: str, fontsize: float, avg_fontsize: float) -> bool:
    line = line.strip()
    if not line:
        return False
    # heading if: larger font OR numbered section like "6.3" OR all caps short line
    if fontsize > avg_fontsize * 1.1:
        return True
    if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):  
        return True
    if line.isupper() and len(line.split()) <= 8:
        return True
    return False

def extract_sections(pdf_bytes: bytes) -> list[dict]:
    sections = []
    current_heading = "General"
    current_content = []

    with pdfplumber.open(pdf_bytes) as pdf:
        # calculate average font size across document
        all_sizes = []
        for page in pdf.pages:
            for char in (page.chars or []):
                if char.get("size"):
                    all_sizes.append(char["size"])
        avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12

        for page_num, page in enumerate(pdf.pages):
            # extract words grouped into lines
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue

            # group words into lines by vertical position
            lines = {}
            for word in words:
                y = round(word["top"])
                if y not in lines:
                    lines[y] = {"text": [], "sizes": []}
                lines[y]["text"].append(word["text"])
                lines[y]["sizes"].append(word.get("size", avg_size))

            for y in sorted(lines.keys()):
                line_text = " ".join(lines[y]["text"])
                line_size = max(lines[y]["sizes"])

                if is_heading(line_text, line_size, avg_size):
                    # save current section before starting new one
                    if current_content:
                        sections.append({
                            "heading": current_heading,
                            "content": " ".join(current_content).strip(),
                            "page":    page_num + 1
                        })
                        current_content = []
                    current_heading = line_text.strip()
                else:
                    current_content.append(line_text)

        # save last section
        if current_content:
            sections.append({
                "heading": current_heading,
                "content": " ".join(current_content).strip(),
                "page":    len(pdf.pages)
            })

    return [s for s in sections if len(s["content"].split()) > 20]


def chunk_section(section: dict, max_words: int=200, overlap: int=20):
    heading = section["heading"]
    content = section["content"]
    words = content.split()
    chunks = []
    i = 0
    
    if len(words) < max_words:
        chunks.append({
            "heading": heading,
            "text": " ".join(words),
            "page": section["page"]
        })
    else:
        while i < len(words):
            chunks.append({
                "heading": heading,
                "text": " ".join(words[i:i+max_words]),
                "page": section["page"]
            })
            i += max_words - overlap
    
    return chunks



def ingest_pdfs():
    files = get_all_files()
    if not files:
        print("No PDFs found in documents table, skipping.")
        return
    # qdrant.delete(
    #     collection_name=COLLECTION_NAME,
    #     points_selector=Filter(
    #         must=[FieldCondition(
    #             key="source_type",
    #             match=MatchValue(value="pdf")
    #         )]
    #     )
    # )


    for f in files:
        name = f["name"]
        pdf_bytes= bytes(f["content"])
        sections=extract_sections(io.BytesIO(pdf_bytes))

        all_chunks = []
        for sec in sections:
            all_chunks.extend(chunk_section(sec))
        

        if not all_chunks:
            print(f"No text extracted from {name}, skipping.")
            continue
        texts=[c["text"] for c in all_chunks]

        embeddings = embedder.encode(texts, show_progress_bar=True)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "source": name,
                    "source_type": "pdf",
                    "heading": chunk["heading"],
                    "doc_id": f["id"]
                }
            )
            for chunk, emb in zip(all_chunks, embeddings)
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
                "text": text,
                "source": f"Clinic policy: {p['topic']}",
                "source_type": "db_policy",
                "policy_id": p["id"]
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
                "text": text,
                "source": f"Drug record: {d['name']}",
                "source_type": "db_drug",
                "drug_id": d["id"]
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
                "text": text,
                "source": f"Lab reference: {l['test_name']}",
                "source_type": "db_lab",
                "lab_id": l["id"]
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

