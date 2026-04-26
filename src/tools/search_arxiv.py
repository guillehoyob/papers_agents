from langchain_core.tools import tool
import httpx, xml.etree.ElementTree as ET
from src.config import settings
import structlog

log = structlog.get_logger()

@tool
def search_arxiv_online(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv. org for recent papers online. Use this ONLY when
    search_local_papers returns insufficient or no results. Returns 
    paper metadata including title, abstract, and URL."""
    try:
        params = {
            "search_query": f"all:{query}",
            "max_results": min(max_results, 10),
            "sortBy": "relevance",
        }
        r = httpx.get(settings.arxiv_base_url, params=params, timeout=20)
        r.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
            papers.append({
                "paper_id": arxiv_id,
                "title": entry.find("atom:title", ns).text.strip(),
                "abstract": entry.find("atom:summary", ns).text. strip()[:800],
                "url": f"https://arxiv.org/abs/{arxiv_id}"
            })
        log.info("search_arxiv", query=query, hits=len(papers))
        return papers
    except Exception as e:
        log.error("search_arxiv_failed", error=str(e))
        return []
