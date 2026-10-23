"""Local retrieval facade. No Chroma instance or embedding model at import."""
from pathlib import Path
from ngc_agent.config import ROOT
from ngc_agent.knowledge import search


def read_model_knowledge():
    path = ROOT / "model_knowledge.md"
    return path.read_text(encoding="utf-8") if path.is_file() else "本地模型知识库为空。"


def search_local_paper_knowledge(query, top_k=4):
    return search(ROOT, query, top_k)


def import_local_pdf(pdf_name):
    from ngc_agent.code import atomic_text
    from pypdf import PdfReader
    directory = (ROOT / "papers").resolve()
    path = (directory / pdf_name).resolve()
    if not path.is_relative_to(directory) or path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError("PDF must be a local file inside papers/")
    reader = PdfReader(path)
    text = "\n\n".join(f"[Page {i+1}]\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages))
    if not text.strip():
        raise ValueError("PDF has no extractable text; OCR is not configured")
    output = ROOT / "papers_extracted" / f"{path.stem}.txt"
    atomic_text(output, text)
    return {"status": "indexed", "source_file": str(output.relative_to(ROOT)), "characters": len(text), "method": "local_lexical"}
