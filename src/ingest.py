import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def ingest_pdf():
    pdf_path = os.getenv("PDF_PATH")
    database_url = os.getenv("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, 
        add_start_index=False).split_documents(docs)
    if not splits:
        raise SystemExit(0)
    
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if isinstance(v, (str, int, float, bool))},
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectors = embeddings.embed_documents([chunk.page_content for chunk in enriched])
    print(f"Chunks: {len(enriched)} | Vetores: {len(vectors)} | Dimensão: {len(vectors[0])}")
    print(f"Amostra (primeiros 5 valores do vetor 0): {vectors[0][:5]}")

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )

    store.add_documents(documents=enriched, ids=ids)
    print(f"Ingested {len(enriched)} chunks into collection '{collection_name}'.")


if __name__ == "__main__":
    ingest_pdf()
