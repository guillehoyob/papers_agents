import chromadb
from langchain_ollama import OllamaEmbeddings
from src.config import settings
from src.ingestion.chunker import Chunk

def get_client():
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)

def get_collection(name: str = "papers"):
    return get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw.space": "cosine"}
    )

embeddings = OllamaEmbeddings(
    base_url=settings.ollama_base_url,
    model=settings.ollama_embed_model,
)

def index_chunks(chunks: list[Chunk], batch: int = 32):
    col = get_collection()
    texts = [c.content for c in chunks]

    all_vecs = []
    for i in range(0, len(texts), batch):
        all_vecs.extend(embeddings.embed_documents(texts[i:i+batch]))
    col.add(
        ids=[f"{c.paper_id}:{c.chunk_index}" for c in chunks],
        documents=texts,
        embeddings=all_vecs,
        metadatas=[
            {"paper_id": c.paper_id, "paper_title": c.paper_title, "chunk_index": c.chunk_index} for c in chunks
        ]
    )

