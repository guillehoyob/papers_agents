# Research Assistant

Two-agent system for academic paper discovery and comparison.

## What it does

A conversational assistant that finds relevant ML/AI papers, reads them,
and answers user questions with proper citations.

## Architecture

[Insert mermaid diagram here — copy from section 01]

### Agents

- **Research Agent**: finds papers via local RAG + arXiv API tool
- **Analysis Agent**: synthesizes answers with inline citations

## Quick start

```
git clone ...
cp .env.example .env  # point OLLAMA_BASE_URL to your instance
python -m scripts.setup_corpus  # downloads 20 papers + indexes
uvicorn src.api.main:app --reload
curl -X POST http://localhost:8000/chat -d '{...}'
```

Or with Docker:

```
docker-compose up --build
```

## Design decisions (honest tradeoffs)

### Why Ollama + Qwen3 instead of OpenAI

- Self-hosted: no API cost, full privacy.
- Tradeoff: slower inference, slightly worse tool calling reliability.
  Mitigated by explicit system prompting.

### Why hybrid retrieval (Chroma + BM25 + reranker) over pure semantic

- Chroma alone misses exact-term matches (acronyms, model names like "LoRA").
- BM25 handles lexical, embeddings handle semantic; RRF fuses rankings without
  manual weight tuning.
- CrossEncoder reranker (bge-reranker-base) reads query+passage jointly —
  much more precise than independent cosine similarity. Biggest single
  quality gain in the pipeline.

### Why ChromaDB over Qdrant

- Zero infra, embedded, perfect for a 20-paper corpus.
- Would switch to Qdrant if scaling past ~100k chunks (and for native hybrid).

### Why LangGraph over raw LangChain

- State is first-class: explicit, debuggable, checkpointable.
- Conditional routing between agents is cleaner.

## Evaluation

Run `python -m evals.run_eval` to score against 10 golden queries.
Current scores: relevance 4.1/5, grounded 3.8/5, clarity 4.3/5.

## What I would do with more time

- Streaming responses via SSE.
- Citation verifier: second LLM pass checking each claim.
- Proper chunking by paper sections (abstract, method, results) instead of
  generic recursive.
- Budget enforcement per request (max tokens, max tool calls).
- Move reranker to GPU for lower latency.
