import nest_asyncio
nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    VectorContextRetriever,
    LLMSynonymRetriever,
)
from config import setup_environment


def main():
    print("\n--- Waking up Hybrid RAG System ---")

    vector_store, graph_store = setup_environment()

    print("Connecting to Vector & Graph Stores...")

    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

    # LLMSynonymRetriever: expands your question into synonyms and
    # searches the graph by the 'name' property on __Entity__ nodes
    llm_retriever = LLMSynonymRetriever(
        index.property_graph_store,
        llm=Settings.llm,
        include_text=True,
        synonym_prompt=None,
        max_keywords=10,
        path_depth=2,
    )

    # VectorContextRetriever: finds nodes by semantic similarity
    # using the embeddings stored during ingestion
    vector_retriever = VectorContextRetriever(
        index.property_graph_store,
        embed_model=Settings.embed_model,
        similarity_top_k=5,
        path_depth=2,
        include_text=True,
    )

    query_engine = index.as_query_engine(
        sub_retrievers=[llm_retriever, vector_retriever],
        llm=Settings.llm,
    )

    # Show what's actually in the graph so you know what to ask about
    print("\n--- Entities currently in your Knowledge Base ---")
    results = graph_store.structured_query(
        """
        MATCH (n:__Entity__)
        RETURN DISTINCT n.name AS name, labels(n) AS type
        ORDER BY type, name
        LIMIT 50
        """
    )
    if results:
        for row in results:
            label = [l for l in row["type"] if l not in ("__Node__", "__Entity__")]
            print(f"  [{', '.join(label)}] {row['name']}")
    else:
        print("  (no entities found)")

    print("\n--- Relationships in your Knowledge Base ---")
    rel_results = graph_store.structured_query(
        """
        MATCH (a:__Entity__)-[r]->(b:__Entity__)
        RETURN a.name AS from, type(r) AS rel, b.name AS to
        LIMIT 30
        """
    )
    if rel_results:
        for row in rel_results:
            print(f"  {row['from']} --[{row['rel']}]--> {row['to']}")
    else:
        print("  (no relationships found)")

    print("\nSystem Ready. Type 'exit' to quit.")
    print("Tip: Ask about the entities listed above!\n")

    while True:
        question = input("Ask your Knowledge Base: ")

        if question.lower() == "exit":
            print("Shutting down. Goodbye!")
            break

        print("Llama 3 is thinking...\n")

        try:
            response = query_engine.query(question)
            answer = str(response).strip()

            if not answer or answer.lower() in ("none", "empty response"):
                print("No answer found. Try asking about one of the entities shown above.\n")
            else:
                print(f"Answer: {answer}\n")

        except Exception as e:
            print(f"Query failed: {str(e)}\n")

        print("-" * 50)


if __name__ == "__main__":
    main()