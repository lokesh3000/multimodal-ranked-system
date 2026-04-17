import pathlib
import sys

from dotenv import load_dotenv

from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document

load_dotenv()

# ---------------------------------------------------------------------------
# Chunking configuration
#
# Long text elements (paragraphs that span half a page or more) are split into
# overlapping windows so that a single dense paragraph doesn't dominate a
# retrieval result and context from surrounding sentences is preserved.
#
# _TEXT_CHUNK_SIZE    — maximum characters per chunk
# _TEXT_CHUNK_OVERLAP — characters shared between adjacent chunks so that
#                       sentences cut at a boundary still appear in both chunks
# Tables and images are never split — they must be stored as atomic units.
# ---------------------------------------------------------------------------
_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long string into overlapping character windows.

    Splitting strategy:
      - Walks through the text in steps of (chunk_size - overlap)
      - Each window is exactly chunk_size characters (or shorter at the end)
      - The overlap ensures sentences cut at a boundary appear in both the
        preceding and following chunk, preserving retrieval context

    This is a lightweight alternative to langchain_text_splitters which is
    not installed in this environment.
    """
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def run_ingestion(file_path: str) -> dict:
    resolved = pathlib.Path(file_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    doc_id = upsert_document(resolved.name, str(resolved))
    print(f"[ingestion] doc_id={doc_id} file={resolved}")

    parsed_elements = parse_document(str(resolved))
    print(f"[ingestion] Docling produced {len(parsed_elements)} elements")

    chunks: list[dict] = []
    for elem in parsed_elements:
        if elem["content_type"] == "text" and len(elem["content"]) > _TEXT_CHUNK_SIZE:
            for sub in _split_text(elem["content"], _TEXT_CHUNK_SIZE, _TEXT_CHUNK_OVERLAP):
                chunks.append({
                    "content": sub,
                    "content_type": elem["content_type"],
                    "metadata": elem["metadata"],
                })
        else:
            chunks.append(elem)

    print(f"[ingestion] {len(chunks)} chunks ready for embedding")
    count = store_chunks(chunks, doc_id)
    print(f"[ingestion] Stored {count} chunks → multimodal_chunks")

    return {"status": "success", "doc_id": doc_id, "chunks_ingested": count}


# ---------------------------------------------------------------------------
# Run ingestion directly:
#   uv run python -m src.ingestion.ingestion
# or from the project root:
#   python src/ingestion/ingestion.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Issue 12 fix: Accept the PDF path as a command-line argument so any
    # document can be ingested without editing the source code.
    # Usage: uv run python -m src.ingestion.ingestion path/to/file.pdf
    # Falls back to the default development PDF when no argument is provided.
    if len(sys.argv) >= 2:
        candidates = [pathlib.Path(arg) for arg in sys.argv[1:]]
    else:
        candidates = [pathlib.Path("data/RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf")]

    paths: list[str] = []
    for candidate in candidates:
        if candidate.is_dir():
            paths.extend(str(p) for p in candidate.glob("*.pdf"))
        else:
            paths.append(str(candidate))

    if not paths:
        raise ValueError("No files found to ingest.")

    results = []
    for file_path in paths:
        results.append(run_ingestion(file_path))

    result = {
        "status": "success",
        "documents": results,
        "total_chunks": sum(item["chunks_ingested"] for item in results),
    }
    print(f"\nIngestion complete: {result}")