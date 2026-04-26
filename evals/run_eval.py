import json, statistics
from pathlib import Path
from langchain_core.messages import HumanMessage
from src.graph import build_graph
from evals.evaluators import judge

app = build_graph(with_memory=False)
golden = json.loads(Path("evals/golden_queries.json").read_text())

results = []
for i, case in enumerate(golden):
    print(f"[{i+1}/{len(golden)}] {case['query'][:60]}")
    state = app.invoke({
        "messages": [HumanMessage(content=case["query"])],
        "papers_found": [], "research_complete": False, "tool_calls_made": [],
    })
    answer = state["messages"][-1].content
    papers = state.get("papers_found", [])

    # heuristics: must_mention check
    mentioned = sum(1 for kw in case["must_mention"] if kw.lower() in answer.lower())
    keyword_score = mentioned / len(case["must_mention"])

    # LLM-as-judge
    score = judge(case["query"], answer, [p.get("paper_title", p.get("title")) for p in papers])
    results.append({
        "query": case["query"],
        "keyword": keyword_score,
        "relevance": score.relevance,
        "grounded": score.grounded,
        "clarity": score.clarity,
        "n_papers": len(papers),
    })

# Aggregate
print("\n=== RESULTS ===")
print(f"Avg keyword coverage: {statistics.mean(r['keyword'] for r in results):.2f}")
print(f"Avg relevance:        {statistics.mean(r['relevance'] for r in results):.2f}/5")
print(f"Avg grounded:         {statistics.mean(r['grounded'] for r in results):.2f}/5")
print(f"Avg clarity:          {statistics.mean(r['clarity'] for r in results):.2f}/5")
Path("evals/results.json").write_text(json.dumps(results, indent=2))