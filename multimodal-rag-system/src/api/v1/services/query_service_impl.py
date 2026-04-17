from typing import Dict, Any

from src.api.v1.agents.agent import run_agent, save_mermaid_image


def query_documents(query: str, k: int = 5, chunk_type: str | None = None) -> Dict[str, Any]:
    result = run_agent(query=query, retrieve_k=k, rerank_k=min(5, k), chunk_type=chunk_type)
    save_mermaid_image()
    return result
