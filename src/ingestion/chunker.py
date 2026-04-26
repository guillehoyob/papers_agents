from pathlib import Path
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

class Chunk(BaseModel):
    content: str
    paper_id: str
    paper_title: str
    chunk_index: int

def extract_text(pdf_path: Path) -> tuple[str, str]:
    """Return (title_guess, full_text). Simple extraction."""
    reader = PdfReader(str(pdf_path))
    pages = [p.extract_text() or "" for p in reader.pages]
    full = "\n".join(pages)
    title = pages[0].strip().split("\n")[0][:200] if pages else pdf_path.stem
    return title, full

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)

def chunk_paper(pdf_path: Path) -> list[Chunk]:
    title, full = extract_text(pdf_path)
    paper_id = pdf_path.stem
    pieces = splitter.split_text(full)
    return [
        Chunk(content=p, paper_id=paper_id, paper_title=title, chunk_index=i) for i, p in enumerate(pieces)
    ]