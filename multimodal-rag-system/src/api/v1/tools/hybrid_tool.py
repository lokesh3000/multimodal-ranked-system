from typing import List, Dict, Any, Optional

from src.api.v1.tools.vector_tool import vector_search
from src.api.v1.tools.fts_tool import fts_search


def hybrid_search(query: str, k: int = 7, chunk_type: Optional[str] = None) -> List[Dict[str, Any]]:
    vector_results = vector_search(query, k=k, chunk_type=chunk_type)
    fts_results = fts_search(query, k=k, chunk_type=chunk_type)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results):
        key = doc["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        chunk_map[key] = doc

    for rank, doc in enumerate(fts_results):
        key = doc["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = doc

    ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [chunk_map[key] for key, _ in ranked[:k]]
