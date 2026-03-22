import nest_asyncio
nest_asyncio.apply()

import chromadb
from llama_index.core import Settings
from llama_index.core import PropertyGraphIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import setup_environment


def get_full_graph_context(graph_store) -> str:
    """Fetch all triples from Neo4j. Used as fallback for conceptual or
    aggregation questions where no specific entity name is matched."""
    triples = graph_store.structured_query(
        """
        MATCH (a:__Entity__)-[r]->(b:__Entity__)
        RETURN a.name AS from_node, type(r) AS relation, b.name AS to_node
        """
    )
    if not triples:
        return ""
    return "\n".join(
        f"{row['from_node']} -> {row['relation']} -> {row['to_node']}"
        for row in triples
    )


def get_context_for_question(question: str, graph_store) -> tuple[str, bool]:
    """
    Returns (context_string, used_full_graph).

    First attempts a targeted keyword match against entity names. If no seed
    nodes are found, falls back to the full graph so the LLM can answer
    conceptual or aggregation questions like 'list all CEOs'.
    """
    words = [w.strip("?.,") for w in question.lower().split() if len(w) > 3]
    conditions = " OR ".join(
        [f"toLower(n.name) CONTAINS '{w}'" for w in words]
    )

    seed_results = graph_store.structured_query(
        f"""
        MATCH (n:__Entity__)
        WHERE {conditions}
        RETURN n.name AS name
        LIMIT 10
        """
    )

    if seed_results:
        seed_names = [r["name"] for r in seed_results]
        triples = graph_store.structured_query(
            """
            MATCH (a:__Entity__)-[r]->(b:__Entity__)
            WHERE a.name IN $names OR b.name IN $names
            RETURN a.name AS from_node, type(r) AS relation, b.name AS to_node
            LIMIT 40
            """,
            param_map={"names": seed_names},
        )
        if triples:
            context = "\n".join(
                f"{row['from_node']} -> {row['relation']} -> {row['to_node']}"
                for row in triples
            )
            return context, False

    # No seed match — fall back to full graph for conceptual questions
    return get_full_graph_context(graph_store), True


def answer_question(question: str, context: str, used_full_graph: bool) -> str:
    scope_note = (
        "The context below contains all known relationships in the knowledge base."
        if used_full_graph
        else "The context below contains relationships relevant to the question."
    )
    prompt = (
        f"You are a knowledge graph assistant. Answer the question using only "
        f"the graph triples provided. Each triple is in the form "
        f"'subject -> relation -> object'.\n"
        f"{scope_note}\n\n"
        f"Graph context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    response = Settings.llm.complete(prompt)
    return response.text.strip()


def main():
    print("\n--- Waking up Hybrid RAG System ---")

    _, graph_store = setup_environment()

    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_collection("hybrid_rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

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
            print(f"  [{', '.join(label) or 'entity'}] {row['name']}")
    else:
        print("  (no entities found -- run ingest_files.py first)")

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

    print("\nSystem Ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask your Knowledge Base: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("Shutting down. Goodbye!")
            break

        print("Thinking...\n")

        try:
            context, used_full_graph = get_context_for_question(question, graph_store)

            if not context:
                print("No data found in the knowledge base.\n")
                print("-" * 50)
                continue

            answer = answer_question(question, context, used_full_graph)

            if not answer or answer.lower() in ("none", "empty response"):
                print("Could not generate an answer. Try rephrasing the question.\n")
            else:
                print(f"Answer: {answer}\n")

        except Exception as e:
            print(f"Query failed: {str(e)}\n")

        print("-" * 50)


if __name__ == "__main__":
    main()
