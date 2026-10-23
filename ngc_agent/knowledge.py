"""Local lexical retrieval. No API, embeddings, model download or database needed."""
from collections import Counter
import math
from pathlib import Path
import re


def tokens(text):
    latin = re.findall(r"[a-z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    return latin + [part[i:i+2] for part in chinese for i in range(max(1, len(part)-1))]


def search(root, query, top_k=4):
    if not query.strip():
        return []
    root = Path(root).resolve()
    paths = [root / "model_knowledge.md"]
    for folder in (root / "knowledge", root / "papers_extracted"):
        if folder.is_dir():
            paths += sorted(path for path in folder.rglob("*") if path.suffix.lower() in {".txt", ".md"} and path.is_file() and path.resolve().is_relative_to(root))
    chunks = []
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for start in range(0, len(text), 900):
            content = text[start:start+1100]
            counts = Counter(tokens(content))
            if counts:
                chunks.append((path, start, content, counts))
    query_tokens = set(tokens(query))
    if not query_tokens or not chunks:
        return []
    document_frequency = Counter(word for _, _, _, counts in chunks for word in counts)
    result = []
    for path, offset, content, counts in chunks:
        length = sum(counts.values())
        score = sum((1 + math.log(counts[word])) * math.log(1 + len(chunks) / (1 + document_frequency[word])) for word in query_tokens if word in counts) / math.sqrt(length)
        if score:
            result.append({"content": content, "source_file": path.relative_to(root).as_posix(),
                           "source_id": f"{path.relative_to(root).as_posix()}:{offset}", "score": round(score, 6), "method": "local_lexical"})
    return sorted(result, key=lambda item: (-item["score"], item["source_id"]))[:min(max(int(top_k), 1), 20)]
