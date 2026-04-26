import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import xml.etree.ElementTree as ET
from pathlib import Path
from src.config import settings

TOPICS = [
    "retrieval augmented generation",
    "LoRA fine-tuning",
    "chain of thought prompting",
    "agent tool use LLM",
]

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = httpx.get(settings.arxiv_base_url, params=params, timeout=30)
    r.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
        papers.append({
            "id": arxiv_id,
            "title": entry.find("atom:title", ns).text.strip(),
            "summary": entry.find("atom:summary", ns).text.strip(),
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        })
    return papers

def download_pdf(url: str, dest: Path):
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)

if __name__ == "__main__":
    # Search and Save Data: Pdf corpus Local diles as inicial DB
    Path(settings.papers_dir).mkdir(parents=True, exist_ok=True)
    all_papers = []
    for topic in TOPICS:
        print(f"Fetching: {topic}")
        all_papers.extend(search_arxiv(topic, max_results=5))

    for p in all_papers:
        dest = Path(settings.papers_dir) / f"{p['id']}.pdf"
        if dest.exists(): continue
        print(f"Downloading {p['id']}...")
        try:
            download_pdf(p["pdf_url"], dest)
        except Exception as e:
            print(f"  failed: {e}")

    # Indexación -- Semántica: ChromaDB
    from src.ingestion.chunker import chunk_paper
    from src.ingestion.embedder import index_chunks

    all_chunks = []
    for pdf in Path(settings.papers_dir).glob("*.pdf"):
        print(f"Chunking {pdf.name}...")
        try:
            all_chunks.extend(chunk_paper(pdf))
        except Exception as e:
            print(f"  failed: {e}")

    print(f"Indexing {len(all_chunks)} chunks...")
    index_chunks(all_chunks)
    print("Done.")

    # Indexación -- Léxica: BM25
    from src.tools.bm25_index import build_and_save

    print("Building BM25 index...")
    build_and_save()
    print("Done.")