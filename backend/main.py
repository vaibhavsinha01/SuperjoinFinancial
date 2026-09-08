import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import store, llm
from backend import embed as embed_mod
from backend.extract import process_pdf
from backend.compare import compare_new_facts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("factloom.main")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="FactLoom")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

store.init_db()


@app.on_event("startup")
def startup():
    store.init_db()
    # Rebuild FAISS index from existing facts so retrieval works after restart
    facts = store.get_all_facts()
    if facts:
        embed_mod.get_faiss_index().rebuild_from_facts(facts)
        logger.info("startup: FAISS index rebuilt with %d facts", embed_mod.get_faiss_index().size)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    llm.reset_metrics()
    new_facts = process_pdf(dest)

    # Insert newly embedded facts into the FAISS index
    for fact in new_facts:
        if fact.get("embedding_status") == "success" and fact.get("embedding"):
            import json
            try:
                vec = json.loads(fact["embedding"]) if isinstance(fact["embedding"], str) else fact["embedding"]
                if vec:
                    embed_mod.get_faiss_index().add(fact["id"], vec)
            except Exception as e:
                logger.warning("could not add fact %s to FAISS: %s", fact.get("id"), e)

    compare_new_facts(new_facts)

    metrics = llm.get_metrics()
    logger.info("upload complete: facts=%d metrics=%s", len(new_facts), metrics)

    return {
        "document_id": new_facts[0]["document_id"] if new_facts else None,
        "facts_extracted": len(new_facts),
        "metrics": metrics,
    }


@app.post("/reprocess")
def reprocess_all():
    """Reprocess all uploaded PDFs and re-run cross-referencing."""
    llm.reset_metrics()
    results = {}
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    for fname in sorted(pdf_files):
        path = os.path.join(UPLOAD_DIR, fname)
        new_facts = process_pdf(path)
        for fact in new_facts:
            if fact.get("embedding_status") == "success" and fact.get("embedding"):
                import json
                try:
                    vec = json.loads(fact["embedding"]) if isinstance(fact["embedding"], str) else fact["embedding"]
                    if vec:
                        embed_mod.get_faiss_index().add(fact["id"], vec)
                except Exception as e:
                    logger.warning("could not add fact %s to FAISS: %s", fact.get("id"), e)
        compare_new_facts(new_facts)
        results[fname] = len(new_facts)

    return {
        "reprocessed": results,
        "total_facts": len(store.get_all_facts()),
        "total_relations": len(store.get_all_relations()),
        "metrics": llm.get_metrics(),
    }


@app.get("/documents")
def list_documents():
    return store.list_documents()


@app.get("/facts")
def get_facts(document_id: str | None = None):
    facts = store.get_facts_by_document(document_id) if document_id else store.get_all_facts()
    for f in facts:
        f.pop("embedding", None)
    return facts


@app.get("/facts/{fact_id}/relations")
def get_fact_relations(fact_id: int):
    fact = store.get_fact(fact_id)
    if not fact:
        raise HTTPException(404, "Fact not found")
    relations = store.get_relations_for_fact(fact_id)
    enriched = []
    for r in relations:
        other_id = r["fact_b_id"] if r["fact_a_id"] == fact_id else r["fact_a_id"]
        other_fact = store.get_fact(other_id)
        if other_fact:
            other_fact.pop("embedding", None)
        enriched.append({**r, "other_fact": other_fact})
    return enriched


@app.get("/relations")
def get_all_relations():
    return store.get_all_relations()


@app.get("/metrics")
def get_metrics():
    """Current session LLM usage metrics."""
    return {
        **llm.get_metrics(),
        "faiss_index_size": embed_mod.get_faiss_index().size,
    }


# Serve simple frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
