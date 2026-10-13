
from src.retrievallab.retrieval import SemanticRetriever


retriever = SemanticRetriever()

while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.strip().lower() == "exit":
        break

    try:
        results = retriever.search(query, top_k=3)

        for rank, result in enumerate(results, start=1):
            print(f"\n{rank}. Score: {result['score']:.3f}")
            print(f"Source: {result['source']}")
            print(result["text"])

    except ValueError as error:
        print(f"Error: {error}")
