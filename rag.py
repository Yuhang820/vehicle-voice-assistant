from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = "data/vectordb"


def simple_splitter(text, chunk_size=200, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )


def build_vectordb():
    print("加载文档...", flush=True)
    with open("data/car_manual.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("切分文档...", flush=True)
    chunk_texts = simple_splitter(text)
    chunks = [Document(page_content=t) for t in chunk_texts if t.strip()]

    print("构建向量库...", flush=True)
    db = Chroma.from_documents(
        chunks,
        get_embeddings(),
        persist_directory=VECTOR_DB_PATH
    )
    print(f"完成，共 {len(chunks)} 个片段", flush=True)
    return db


def load_vectordb():
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embeddings()
    )


def query_manual(question: str, k: int = 2) -> str:
    if not os.path.exists(VECTOR_DB_PATH):
        build_vectordb()
    db = load_vectordb()
    results = db.similarity_search(question, k=k)
    return "\n".join([r.page_content for r in results])


if __name__ == "__main__":
    build_vectordb()