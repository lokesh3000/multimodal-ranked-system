from typing import List, Dict, Any, Optional

from src.core.db import get_db_conn


def fts_search(query: str, k: int = 7, chunk_type: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = """
        SELECT
            content,
            chunk_type,
            page_number,
            section,
            source_file,
            element_type,
            metadata,
            ts_rank(to_tsvector('english', content), plainto_tsquery('english', %(query)s)) AS fts_rank
        FROM multimodal_chunks
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %(query)s)
    """
    if chunk_type:
        sql += "\n            AND chunk_type = %(chunk_type)s"
    sql += "\n        ORDER BY fts_rank DESC\n        LIMIT %(k)s;"

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "query": query,
                "chunk_type": chunk_type,
                "k": k,
            })
            rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        results.append({
            "rank": rank,
            "similarity": float(row["fts_rank"] or 0.0),
            "content": row["content"],
            "chunk_type": row["chunk_type"],
            "metadata": {
                "page_number": row.get("page_number"),
                "section": row.get("section"),
                "source_file": row.get("source_file"),
                "element_type": row.get("element_type"),
            },
        })

    return results
