import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.db import similarity_search

load_dotenv()

_SYSTEM_PROMPT = (
    "You are a helpful assistant for document question-answering. "
    "Answer the question using ONLY the provided context (text, tables, and images). "
    "If the answer is not present in the context, say you don't know. "
    "When citing information, mention the page number and section."
)

# Issue 8 fix: Module-level LLM singleton — avoids re-instantiating a new HTTP
# client on every request.
_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_LLM_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def query_documents(query: str, k: int = 5, chunk_type: str | None = None) -> dict:
    # ── Step 1: Retrieve top-k relevant chunks ────────────────────────────────────────
    chunks = similarity_search(query, k=k, chunk_type=chunk_type)  # Issue 17

    # ── Step 2: Build multimodal message content ──────────────────────────────
    # Text and table chunks become plain-text context blocks.
    # Image chunks that have actual bytes are inlined as base64 data URIs
    # so Gemini Vision can reason over them directly.
    message_parts: list[dict] = []
    sources: list[dict] = []

    text_blocks: list[str] = []
    for chunk in chunks:
        chunk_type = chunk["chunk_type"]
        page = chunk.get("page_number")
        section = chunk.get("section") or "—"
        source_file = chunk.get("source_file", "")

        sources.append({
            "chunk_type": chunk_type,
            "page_number": page,
            "section": section,
            "source_file": source_file,
            "element_type": chunk.get("element_type"),
            "similarity": round(chunk.get("similarity", 0), 4),
        })

        if chunk_type in ("text", "table"):
            label = "TABLE" if chunk_type == "table" else "TEXT"
            text_blocks.append(
                f"[{label} | page {page} | {section}]\n{chunk['content']}"
            )
        elif chunk_type == "image" and chunk.get("image_base64"):
            # Flush accumulated text before inserting an image part
            if text_blocks:
                message_parts.append({
                    "type": "text",
                    "text": "\n\n".join(text_blocks),
                })
                text_blocks = []
            message_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{chunk['image_base64']}"
                },
            })


    # Flush any remaining text blocks
    if text_blocks:
        message_parts.append({
            "type": "text",
            "text": "\n\n".join(text_blocks),
        })

    print("************");
    print(text_blocks)
    print("************");

    # Append the actual question at the end
    message_parts.append({
        "type": "text",
        "text": f"\n\nQuestion: {query}",
    })

    # ── Step 3: Invoke Gemini ──────────────────────────────────────────────────
    # ---------------------------------------------------------------------------
    # Issue 11 — SHORT-TERM FIX (Gemini Vision description at ingestion time):
    # During ingestion, call Gemini Vision on each image to generate a rich
    # textual description and embed THAT instead of just the caption.
    # Add to docling_parser.py inside the picture branch:
    #
    # def _describe_image_with_gemini(img_b64: str, api_key: str) -> str:
    #     from langchain_core.messages import HumanMessage
    #     from langchain_google_genai import ChatGoogleGenerativeAI
    #     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    #     msg = HumanMessage(content=[
    #         {"type": "text",
    #          "text": "Describe this image in detail for search indexing. "
    #                  "Include any numbers, labels, or visual patterns."},
    #         {"type": "image_url",
    #          "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
    #     ])
    #     return llm.invoke([msg]).content
    #
    # Replace:
    #   content = caption.strip() or f"[Image on page {page_no}]"
    # With:
    #   content = (
    #       _describe_image_with_gemini(img_b64, os.getenv("GOOGLE_API_KEY"))
    #       if img_b64 else (caption.strip() or f"[Image on page {page_no}]")
    #   )
    # ---------------------------------------------------------------------------
    # Issue 11 — LONG-TERM FIX (gemini-embedding-2-preview multimodal embeddings):
    # Use Google's gemini-embedding-2-preview model which embeds both text AND
    # image bytes directly into the same vector space, enabling true visual
    # similarity search without requiring text captions.
    # In db.py store_chunks(), for image chunks, replace the caption embedding
    # with a multimodal embedding using the google-genai SDK:
    #
    # from google import genai
    # from google.genai import types as genai_types
    # client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # result = client.models.embed_content(
    #     model="gemini-embedding-2-preview",
    #     contents=genai_types.Content(
    #         parts=[genai_types.Part.from_bytes(
    #             data=base64.b64decode(img_b64), mime_type="image/png"
    #         )]
    #     ),
    # )
    # image_embedding = result.embeddings[0].values
    # ---------------------------------------------------------------------------

    # Issue 8 fix: Use the module-level singleton instead of creating a new client.
    # Issue 7 fix: Use SystemMessage so the system prompt is sent in the correct role.
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=message_parts),
    ]

    response = _llm.invoke(messages)  # Issue 8: use singleton

    # response.content may be a plain string (most models) or a list of
    # content parts (thinking models like gemini-3.1-pro-preview).
    # Extract just the text in either case.
    content = response.content
    if isinstance(content, list):
        answer = " ".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        answer = content

    return {
        "answer": answer,
        "sources": sources,
    }
