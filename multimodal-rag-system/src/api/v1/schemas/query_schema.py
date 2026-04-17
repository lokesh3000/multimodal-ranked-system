from pydantic import BaseModel, Field
from typing import List, Optional

# ---- Request ----
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    chunk_type: Optional[str] = Field(
        None, description="Filter by content type: 'text', 'table', or 'image'"
    )

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    selected_tool: str
    validated: bool
    retry_count: int