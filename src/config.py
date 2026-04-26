from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str  = "http://localhsot:11434"
    ollama_chat_model: str = "qwen3:14b"
    ollama_embed_model: str = "qwen3-embedding:4b"
    chroma_persist_dir: str = "./data/chroma_db"
    papers_dir: str = "./data/raw"
    checkpoint_db_path: str = "./data/checkpoints.sqlite"
    arxiv_base_url:str = "https://export.arxiv.org/api/query"

settings = Settings()
