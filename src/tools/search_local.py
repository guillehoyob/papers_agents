from langchain_core.tools import tool
from src.ingestion.embedder import get_collection, embeddings
from src.tools.bm25_index import load as load_bm25
from src.tools.reranker import rerank
import structlog

log = structlog.get_logger()

def _semantic(query: str, top_k: int = 20) -> list[dict]:
    col = get_collection()
    qvec = embeddings.embed_query(query)
    r = col.query(query_embeddings=[qvec], n_results=top_k)
    out = []
    for i, (id_, d, m, dist) in enumerate(zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])):
        out.append({"id": id_, "passage": d, "meta": m, "rank": i})
    return out

def _reciprocal_rank_fusion(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """RRF: score = sum(1 / (k + rank_in_list_i)). Independent of absolute scores."""
    fused = {}
    for results in lists:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            if doc_id not in fused:
                fused[doc_id] = {**item, "rrf_score": 0.0}
            fused[doc_id]["rrf_score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda x: -x["rrf_score"])

@tool
def search_local_papers(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid search over the local paper corpus: semantic (embeddings) +
    keyword (BM25) fused with RRF, then reranked with a cross-encoder.
    Use this FIRST for any research question. Returns top passages with
    paper title and id for citation."""
    try:
        sem = _semantic(query, top_k=20)
        lex_raw = load_bm25().search(query, top_k=20)
        # normalize BM25 output shape
        lex = [{"id": r["id"], "passage": r["passage"], "meta": r["meta"]} for r in lex_raw]
    
        fused = _reciprocal_rank_fusion([sem, lex])[:10]
        reranked = rerank(query, fused, top_k=top_k)

        out = [{
            "passage": c["passage"],
            "paper_id": c["meta"]["paper_id"],
            "paper_title": c["meta"]["paper_title"],
            "score": round(c["rerank_score"], 3),
        } for c in reranked if c["rerank_score"] > -5]

        log.info("search_local", query=query, sem=len(sem), lex=len(lex), final=len(out))
        return out
    except Exception as e:
        log.error("search_local_failed", error=str(e))
        return []

