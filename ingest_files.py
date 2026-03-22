import os
import logging
import chromadb
from llama_index.core import Settings
from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    StorageContext,
)
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import setup_environment

logging.basicConfig(level=logging.INFO)

def run_multi_file_ingestion():
    print("--- Building Hybrid Knowledge Base ---")

    _, graph_store = setup_environment()

    # Wipe Neo4j
    print("Clearing previous graph data...")
    try:
        graph_store.structured_query("MATCH (n) DETACH DELETE n")
        print("Graph cleared.")
    except Exception:
        print("Could not clear graph. Continuing...")

    # Wipe and recreate Chroma
    print("Resetting Chroma collection...")
    db = chromadb.PersistentClient(path="./chroma_db")
    try:
        db.delete_collection("hybrid_rag_collection")
    except Exception:
        pass
    fresh_collection = db.create_collection("hybrid_rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=fresh_collection)
    print("Chroma collection ready.")

    # Load documents
    print(f"\nScanning directory: {os.path.abspath('data')}")
    reader = SimpleDirectoryReader(input_dir="./data")
    documents = reader.load_data()
    print(f"Found {len(documents)} document(s). Starting extraction...")

    kg_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=10,
        num_workers=1,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        graph_store=graph_store,
    )

    print("\nExtracting relationships and building index...")
    print("This may take several minutes depending on corpus size.\n")

    try:
        index = PropertyGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            property_graph_store=graph_store,
            vector_store=vector_store,
            kg_extractors=[kg_extractor],
            embed_model=Settings.embed_model,
            embed_kg_nodes=True,
            show_progress=True,
        )

        # embed_kg_nodes=True can silently skip writing embeddings onto Neo4j
        # nodes in some LlamaIndex versions. As a guaranteed fallback, we
        # read every entity node back from Neo4j and set the embedding
        # property manually using the same embed_model used during ingestion.
        print("\nVerifying node embedding coverage...")
        emb_check = graph_store.structured_query(
            "MATCH (n:__Entity__) RETURN count(n) AS total, count(n.embedding) AS with_embedding"
        )
        total = emb_check[0]["total"] if emb_check else 0
        with_emb = emb_check[0]["with_embedding"] if emb_check else 0

        if total > 0 and with_emb == 0:
            print(f"embed_kg_nodes did not persist embeddings ({total} nodes affected).")
            print("Writing embeddings manually...")

            name_results = graph_store.structured_query(
                "MATCH (n:__Entity__) WHERE n.name IS NOT NULL RETURN n.name AS name"
            )
            names = [r["name"] for r in name_results if r.get("name")]

            # Embed in small batches to avoid Ollama timeouts
            batch_size = 5
            written = 0
            for i in range(0, len(names), batch_size):
                batch = names[i: i + batch_size]
                embeddings = Settings.embed_model.get_text_embedding_batch(batch)
                for name, emb in zip(batch, embeddings):
                    graph_store.structured_query(
                        "MATCH (n:__Entity__ {name: $name}) SET n.embedding = $emb",
                        param_map={"name": name, "emb": emb},
                    )
                    written += 1
                print(f"  Embedded {written}/{len(names)} nodes...")
            
            print(f"Embeddings written for {written} nodes.")
        else:
            print(f"Embeddings present on {with_emb}/{total} nodes.")

        # Final sanity check
        results = graph_store.structured_query(
            "MATCH (n:__Entity__) RETURN count(n) AS total, count(n.embedding) AS with_embedding"
        )
        total = results[0]["total"] if results else 0
        with_emb = results[0]["with_embedding"] if results else 0

        print("\n" + "=" * 40)
        print("INGESTION COMPLETE")
        print(f"  Entities in Neo4j       : {total}")
        print(f"  Entities with embedding : {with_emb}")
        print(f"  Vectors in Chroma       : {fresh_collection.count()}")
        print(f"  Graph store             : bolt://localhost:7687")
        print(f"  Vector store            : ./chroma_db")
        print("=" * 40)

        if with_emb < total:
            print(f"\nWARNING: {total - with_emb} nodes are still missing embeddings.")
            print("query.py will use keyword-only retrieval as fallback for those nodes.")

    except Exception as e:
        print(f"\nIngestion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. docker ps       -- confirm Neo4j and Ollama are running")
        print("  2. ollama list     -- confirm llama3 and nomic-embed-text are pulled")
        print("  3. Check virtual environment is active")

if __name__ == "__main__":
    run_multi_file_ingestion()
