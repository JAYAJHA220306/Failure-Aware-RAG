"""Load documents (.md, .txt, .pdf), split them into overlapping chunks, and save the chunk store.

Optional front-matter at the top of .md/.txt files sets metadata used by repairs:

    ---
    title: Hypoglycaemia management (demo)
    authority: guideline        # guideline | textbook | article | forum
    ---
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import settings

AUTHORITY_RANK = {"guideline": 3, "textbook": 2, "article": 1, "forum": 0}
SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    authority: str = "article"
    position: int = 0

    @property
    def authority_rank(self) -> int:
        return AUTHORITY_RANK.get(self.authority, 1)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.split("#")[0].strip()
            text = text[end + 4 :]
    return meta, text


def read_document(path: Path) -> tuple[dict, str]:
    if path.suffix.lower() == ".pdf":
        return {}, _read_pdf(path)
    return _parse_front_matter(path.read_text(encoding="utf-8", errors="ignore"))


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into ~size-word windows; long paragraphs are split with overlap."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block.startswith("#"):  # a markdown heading becomes a short sentence
            block = re.sub(r"^#+\s*", "", block).rstrip(".") + "."
        block = re.sub(r"\s+", " ", block).strip()
        if block and block != ".":
            paragraphs.append(block)
    chunks: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) > size:
            if current:
                chunks.append(" ".join(current))
                current = []
            step = max(1, size - overlap)
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i : i + size]))
                if i + size >= len(words):
                    break
            continue
        if len(current) + len(words) > size and current:
            chunks.append(" ".join(current))
            current = []  # paragraphs are natural boundaries; overlap only inside long paragraphs
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(path: Path) -> list[Chunk]:
    meta, text = read_document(path)
    doc_id = path.stem
    title = meta.get("title") or path.stem.replace("_", " ").replace("-", " ").title()
    guessed = next((a for a in AUTHORITY_RANK if a in path.stem.lower()), "article")  # e.g. nice_guideline.pdf
    authority = meta.get("authority", guessed).lower()
    pieces = chunk_text(text, settings.chunk_words, settings.chunk_overlap)
    return [
        Chunk(id=f"{doc_id}#{i}", doc_id=doc_id, title=title, text=p, authority=authority, position=i)
        for i, p in enumerate(pieces)
    ]


def build_store(corpus_dir: Path | None = None, index_dir: Path | None = None) -> list[Chunk]:
    """Chunk every supported file in corpus_dir and write chunks.jsonl to index_dir."""
    corpus_dir = corpus_dir or settings.corpus_dir
    index_dir = index_dir or settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED and not path.name.lower().startswith("readme"):
            chunks.extend(chunk_document(path))
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    # Embeddings depend on the chunks, so drop any stale cache.
    for stale in index_dir.glob("embeddings*.npy"):
        stale.unlink()
    return chunks


def load_store(index_dir: Path | None = None) -> list[Chunk]:
    index_dir = index_dir or settings.index_dir
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return build_store(index_dir=index_dir)
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


if __name__ == "__main__":
    built = build_store()
    print(f"Indexed {len(built)} chunks from {settings.corpus_dir}")
