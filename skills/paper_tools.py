"""Explicit online arXiv tools, loaded only when requested."""
import re
from ngc_agent.config import ROOT


def search_arxiv_paper(topic, max_count=5):
    import arxiv
    if not topic.strip():
        raise ValueError("Search topic cannot be empty")
    count = min(max(int(max_count), 1), 10)
    client = arxiv.Client(page_size=count, delay_seconds=3, num_retries=2)
    results = client.results(arxiv.Search(query=topic, max_results=count, sort_by=arxiv.SortCriterion.Relevance))
    return [{"title": item.title, "summary": item.summary, "abstract": item.summary, "arxiv_id": item.get_short_id(),
             "url": item.entry_id, "pdf_url": item.pdf_url,
             "authors": [author.name for author in item.authors], "published": item.published.isoformat(), "publish_date": item.published.date().isoformat()}
            for item in results]


def download_arxiv_pdf(arxiv_id):
    import arxiv
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", arxiv_id):
        raise ValueError("Invalid arXiv ID")
    item = next(arxiv.Client(num_retries=2).results(arxiv.Search(id_list=[arxiv_id])), None)
    if item is None:
        raise ValueError("arXiv paper not found")
    directory = ROOT / "papers"
    directory.mkdir(exist_ok=True)
    filename = arxiv_id.replace("/", "_") + ".pdf"
    item.download_pdf(dirpath=str(directory), filename=filename)
    return {"status": "downloaded", "path": str(directory / filename), "indexed": False,
            "note": "Run import_local_pdf explicitly to extract local text; no embedding model is downloaded."}
