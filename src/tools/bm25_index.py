import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
import pickle, re
from rank_bm25 import BM25Okapi
from src.config import settings
from src.ingestion.embedder import get_collection

BM25_PATH = Path(settings.chroma_persist_dir).parent / "bm25.pkl"

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

class BM25Store:
    def __init__(self, ids, docs, metas):
        self.ids = ids
        self.docs = docs
        self.metas = metas
        self.bm25 = BM25Okapi([_tokenize(d) for d in docs])

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return [
            {"id": self.ids[i], "passage": self.docs[i], "meta": self.metas[i], "score": float(s)} 
            for i, s in ranked if s > 0
        ]

def build_and_save():
    col = get_collection()
    data = col.get()
    store = BM25Store(data["ids"], data["documents"], data["metadatas"])
    #BM25_PATH.parent.mkdir(parents=True, exist_ok=True)   
    with open(BM25_PATH, "wb") as f:
        pickle.dump(store, f)
    return store

_cache = {"store": None}
def load() -> BM25Store:
    if _cache["store"] is None:
        if BM25_PATH.exists():
            with open(BM25_PATH, "rb") as f:
                _cache["store"] = pickle.load(f)
        else:
            _cache["store"] = build_and_save()
    return _cache["store"]