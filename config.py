import chromadb
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

def setup_environment():
    print("--- Connecting to Local AI Infrastructure (Ollama Edition) ---")

    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        request_timeout=300.0
    )

    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    Settings.chunk_size = 1024       # was 512
    Settings.chunk_overlap = 100     # slightly wider overlap avoids missing

    Settings.embed_batch_size = 1

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("hybrid_rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    graph_store = Neo4jPropertyGraphStore(
        username="neo4j",
        password="password123",
        url="bolt://localhost:7687",
        database="neo4j",
    )

    print("Connection Successful: 100% Dockerized AI Stack Ready.")

    return vector_store, graph_store

if __name__ == "__main__":
    setup_environment()
