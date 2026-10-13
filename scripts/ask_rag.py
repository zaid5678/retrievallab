
from src.retrievallab.rag import RAGPipeline


rag = RAGPipeline()

print("RetrievalLab RAG — type 'exit' to quit.")

while True:
    question = input("\nQuestion: ").strip()

    if question.lower() == "exit":
        break

    try:
        result = rag.answer(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nRetrieved sources:")
        for item in result["sources"]:
            print(
                f"- {item['source']} "
                f"(score: {item['score']:.3f})"
            )

    except ValueError as error:
        print(f"Error: {error}")

    except Exception as error:
        print(f"Unexpected error: {error}")
