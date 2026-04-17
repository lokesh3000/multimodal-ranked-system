from typing import List, Dict, Any, Optional

from src.core.db import similarity_search


def vector_search(query: str, k: int = 7, chunk_type: Optional[str] = None) -> List[Dict[str, Any]]:
    results = similarity_search(query, k=k, chunk_type=chunk_type)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
        row["metadata"] = {
            "page_number": row.get("page_number"),
            "section": row.get("section"),
            "source_file": row.get("source_file"),
            "element_type": row.get("element_type"),
        }
    return results
