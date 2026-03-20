import os
import logging
from llama_index.core import Settings
from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    Settings,
    StorageContext
)
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from config import setup_environment

# Set up logging so you can see the LLM's extraction progress in the terminal
logging.basicConfig(level=logging.INFO)

def run_multi_file_ingestion():
    print("--- Building Hybrid Knowledge Base ---")

    # 1. Connect to our local infrastructure
    vector_store, graph_store = setup_environment()

    # 2. Load ALL documents from the /data folder
    # SimpleDirectoryReader automatically picks up investigation.txt, jwst_mission.txt, etc.
    print(f"Scanning directory: {os.path.abspath('data')}")
    reader = SimpleDirectoryReader(input_dir="./data")
    documents = reader.load_data()

    print(f"Found {len(documents)} documents. Starting extraction...")

    # 3. Configure the KG extractor
    kg_extractor = DynamicLLMPathExtractor(
        llm=Settings.llm,
        max_triplets_per_chunk=15,  # Find up to 15 connections per chunk
        allowed_entity_types=None,  # LLM decides the entity types
        allowed_relation_types=None, # LLM decides the relationship types
    )

    # 4. Configure StorageContext with both stores
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        graph_store=graph_store
    )

    print("\nLlama 3 is reading and connecting the dots...")
    print("This will take a few minutes as we are extracting complex relationships.\n")

    try:
        # 5. Build the Property Graph Index
        # This is the 'heavy lifting' step where the LLM reads every sentence.
        index = PropertyGraphIndex.from_documents(
            documents,
            storage_context=storage_context,       # <-- FIX: pass storage_context
            property_graph_store=graph_store,
            vector_store=vector_store,
            kg_extractors=[kg_extractor],
            embed_model=Settings.embed_model,
            show_progress=True
        )

        print("\n" + "="*40)
        print("INGESTION SUCCESSFUL")
        print(f"Vectors saved to: ./chroma_db")
        print(f"Graph mapped to:  bolt://localhost:7687")
        print("="*40)

    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        print("\nTroubleshooting Tips:")
        print("1. Are your Docker containers (Neo4j & Ollama) actually running?")
        print("2. Did you pull the model inside the Ollama container?")
        print("3. Is your virtual environment active?")

if __name__ == "__main__":
    run_multi_file_ingestion()