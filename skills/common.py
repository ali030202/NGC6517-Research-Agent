"""Optional PDF/vector helpers retained for external extensions."""
_embed_models = {}


def get_embed_model(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    if model_name not in _embed_models:
        # Explicitly installed local models only. Core retrieval uses no embeddings.
        _embed_models[model_name] = SentenceTransformer(model_name, local_files_only=True)
    return _embed_models[model_name]


def add_pdf_to_vector_db(pdf_path, collection, embed_model):
    from pathlib import Path
    import hashlib
    from pypdf import PdfReader
    path = Path(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    chunks = [text[start:start+1000] for start in range(0, len(text), 800) if text[start:start+1000].strip()]
    if not chunks:
        raise ValueError("PDF contains no text")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    embeddings = embed_model.encode(chunks).tolist()
    collection.upsert(ids=[f"{checksum}:{i}" for i in range(len(chunks))], documents=chunks,
                      embeddings=embeddings, metadatas=[{"source_file": path.name}] * len(chunks))
    return {"chunks": len(chunks), "sha256": checksum}
