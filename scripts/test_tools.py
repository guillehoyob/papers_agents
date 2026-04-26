import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.search_local import search_local_papers
from src.tools.search_arxiv import search_arxiv_online

print("=== LOCAL ===")
for r in search_local_papers.invoke({"query": "LoRA low rank adaptation", "top_k": 3}):
    print(f"  [{r['score']}] {r['paper_title'][:60]}")

print("\n=== ARXIV ===")
for r in search_arxiv_online.invoke({"query": "mixture of experts transformers", "max_results": 3}):
    print(f"  {r['title'][:60]}")