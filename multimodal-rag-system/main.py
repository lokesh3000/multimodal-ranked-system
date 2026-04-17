from fastapi import FastAPI
from src.api.v1.routes.query import router

def main():
    print("Hello from multimodal-rag-system!")

app = FastAPI(title="RAG API")

# we will enable rest api endpoint at localhost:8000/
# @app.get("/query")
# def read_root():
#     return {
#         "status": "query received"
#     }

# # health check endpoint
# @app.get("/admin/upload")
# def health_check():
#     return {
#         "status": "ok"
#     }

app.include_router(router, prefix="/api/v1")
app.include_router(router, prefix="/api/v1/admin")

if __name__ == "__main__":
    main()
