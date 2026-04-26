from sentence_transformers import CrossEncoder
import structlog
import time
import torch

log = structlog.get_logger()
device  ="cuda" if torch.cuda.is_available() else "cpu"


_model = {"m": None}
def get_reranker():
    if _model["m"] is None:
        log.info("loading_reranker")
        _model["m"] = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device=device)  #bge-reranker-base
    log.info("reranker_loaded", device=device)
    return _model["m"]

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """candidates: lsit of dicts with 'passage' key. Returns sorted by rerank score."""
    if not candidates:
        return[]
    model = get_reranker()
    pairs = [(query, c["passage"]) for c in candidates]

    t0 = time.time()
    scores = model.predict(pairs)
    log.info("rerank_done", n=len(pairs), elapsed=time.time() - t0)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]