import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel

from src.api.v1.tools.vector_tool import vector_search
from src.api.v1.tools.fts_tool import fts_search
from src.api.v1.tools.hybrid_tool import hybrid_search

load_dotenv(override=True)


class AgentState(TypedDict):
    query: str
    selected_tool: str
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    answer: str
    validated: bool
    retry_count: int
    retrieve_k: int
    rerank_k: int
    chunk_type: Optional[str]


class ToolChoice(BaseModel):
    tool: Literal["vector", "fts", "hybrid"]


class ValidationResponse(BaseModel):
    validated: bool


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "google_genai:gemini-3.1-flash-lite-preview"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )


def choose_tool_node(state: AgentState) -> AgentState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(ToolChoice)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a retrieval tool selector. Choose exactly one tool: vector, fts, or hybrid. "
            "Use fts when the user query contains IDs, policy numbers, abbreviations, or exact-keyword lookup terms. "
            "Use hybrid for short queries with mixed semantic and keyword intent. "
            "Use vector for open-ended, conceptual, or descriptive questions."
        ),
        (
            "human",
            "Question: {query}\n\nRespond only with one tool name: vector, fts, or hybrid."
        )
    ])

    try:
        result = (prompt | structured_llm).invoke({"query": state["query"]})
        tool = result.model_dump().get("tool", "vector")
    except Exception:
        tool = "vector"

    tool = tool.lower().strip()
    if tool not in {"vector", "fts", "hybrid"}:
        if len(state["query"].split()) <= 3:
            tool = "hybrid"
        elif any(token.isupper() and token.isalnum() for token in state["query"].split()):
            tool = "fts"
        else:
            tool = "vector"

    print(f"[choose_tool_node] Selected tool: {tool}")
    return {**state, "selected_tool": tool}


def retriever_node(state: AgentState) -> AgentState:
    if state["selected_tool"] == "fts":
        docs = fts_search(state["query"], k=state["retrieve_k"], chunk_type=state["chunk_type"])
    elif state["selected_tool"] == "hybrid":
        docs = hybrid_search(state["query"], k=state["retrieve_k"], chunk_type=state["chunk_type"])
    else:
        docs = vector_search(state["query"], k=state["retrieve_k"], chunk_type=state["chunk_type"])

    if not docs:
        print("[retriever_node] No docs found for selected tool, falling back to vector search")
        docs = vector_search(state["query"], k=state["retrieve_k"], chunk_type=state["chunk_type"])

    print(f"[retriever_node] Retrieved {len(docs)} docs via {state['selected_tool']}")
    return {**state, "retrieved_docs": docs}


def rerank_node(state: AgentState) -> AgentState:
    docs = state["retrieved_docs"]
    if not docs:
        print("[rerank_node] No docs to rerank")
        return {**state, "reranked_docs": []}

    reranked = sorted(
        docs,
        key=lambda doc: doc.get("similarity", doc.get("rank", 0)),
        reverse=True,
    )[: state["rerank_k"]]

    for rank, doc in enumerate(reranked, start=1):
        doc["rank"] = rank

    print(f"[rerank_node] Reranked to top {len(reranked)} docs")
    return {**state, "reranked_docs": reranked}


def generate_answer_node(state: AgentState) -> AgentState:
    if not state["reranked_docs"]:
        fallback = "I don't have enough information to answer that question with confidence."
        return {**state, "answer": fallback}

    llm = get_llm()
    context = "\n\n".join(
        f"[Source: {doc['metadata'].get('source_file', 'unknown')} | page {doc['metadata'].get('page_number', '?')} | section {doc['metadata'].get('section', '—')} ]\n{doc['content']}"
        for doc in state["reranked_docs"]
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful document assistant. Answer the user's question directly using only the provided context. "
            "If the answer is not present, say 'I don't have that information at the moment.' "
            "Cite the source page and section at the end of your answer."
        ),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    result = (prompt | llm).invoke({"context": context, "query": state["query"]})
    answer = result.model_dump().get("content", "I don't have that information at the moment.") if hasattr(result, "model_dump") else str(result)

    if isinstance(answer, list):
        answer = " ".join(
            part.get("text", "") for part in answer if isinstance(part, dict)
        )

    print("[generate_answer_node] Answer generated")
    return {**state, "answer": answer}


def validate_node(state: AgentState) -> AgentState:
    if not state["reranked_docs"]:
        return {**state, "validated": False, "retry_count": state["retry_count"] + 1}

    llm = get_llm()
    structured_llm = llm.with_structured_output(ValidationResponse)
    context = "\n\n".join(
        f"[Source: {doc['metadata'].get('source_file', 'unknown')} | page {doc['metadata'].get('page_number', '?')} | section {doc['metadata'].get('section', '—')} ]\n{doc['content']}"
        for doc in state["reranked_docs"]
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a validation assistant. Determine whether the provided answer is fully supported by the retrieved documents. "
            "Respond only with a single boolean field named validated."
        ),
        (
            "human",
            "Question: {query}\nAnswer: {answer}\n\nSupporting documents:\n{context}\n\n"
            "Is the answer supported by the documents?"
        )
    ])

    try:
        result = (prompt | structured_llm).invoke({
            "query": state["query"],
            "answer": state["answer"],
            "context": context,
        })
        validated = result.model_dump().get("validated", False)
    except Exception as exc:
        print(f"[validate_node] validation failed: {exc}")
        validated = False

    retry_count = state["retry_count"]
    if not validated:
        retry_count += 1

    print(f"[validate_node] validated={validated}, retry_count={retry_count}")
    return {**state, "validated": validated, "retry_count": retry_count}


def output_node(state: AgentState) -> AgentState:
    return state


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("tool_selector", choose_tool_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", rerank_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("validate", validate_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("tool_selector")
    graph.add_edge("tool_selector", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generate_answer")
    graph.add_edge("generate_answer", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda state: "output" if state["validated"] or state["retry_count"] >= 3 else "retriever",
    )
    graph.set_finish_point("output")

    return graph.compile()


agent_graph = build_agent_graph()


def run_agent(query: str, retrieve_k: int = 7, rerank_k: int = 5, chunk_type: Optional[str] = None) -> Dict[str, Any]:
    initial_state: AgentState = {
        "query": query,
        "selected_tool": "vector",
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "validated": False,
        "retry_count": 0,
        "retrieve_k": retrieve_k,
        "rerank_k": rerank_k,
        "chunk_type": chunk_type,
    }
    final_state = agent_graph.invoke(initial_state)
    return {
        "answer": final_state.get("answer", ""),
        "selected_tool": final_state.get("selected_tool", "vector"),
        "validated": final_state.get("validated", False),
        "retry_count": final_state.get("retry_count", 0),
        "sources": [
            {
                "page_number": doc["metadata"].get("page_number"),
                "section": doc["metadata"].get("section"),
                "source_file": doc["metadata"].get("source_file"),
                "element_type": doc["metadata"].get("element_type"),
                "similarity": round(doc.get("similarity", doc.get("rank", 0)), 4),
            }
            for doc in final_state.get("reranked_docs", [])
        ],
    }


def save_mermaid_image(path: str = "data/agent_workflow.png") -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    graph = agent_graph.get_graph()
    image_bytes = graph.draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER
    )
    path_obj.write_bytes(image_bytes)

    print(f"[save_mermaid_image] Mermaid workflow image written to {path_obj}")
    return str(path_obj.resolve())


if __name__ == "__main__":
    save_mermaid_image()
