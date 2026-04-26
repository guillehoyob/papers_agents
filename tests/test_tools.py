from src.tools.search_local import search_local_papers

def test_search_local_returns_list():
    r = search_local_papers.invoke({"query": "transformer", "top_k": 3})
    assert isinstance(r, list)
    assert len(r) <= 3
    for item in r:
        assert "paper_id" in item
        assert "passage" in item

def test_search_local_empty_query_does_not_crash():
    r = search_local_papers.invoke({"query": "", "top_k": 3})
    assert isinstance(r, list)  # puede estar vacío, pero no crash