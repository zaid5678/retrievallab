
from src.retrievallab.chunking import load_documents, chunk_documents

documents = load_documents()
chunks = chunk_documents(documents, chunk_size=50, overlap=10)

print(f"Loaded {len(documents)} documents")
print(f"Created {len(chunks)} chunks\n")

for chunk in chunks:
    print(f"ID: {chunk['chunk_id']}")
    print(f"Source: {chunk['source']}")
    print(f"Words: {len(chunk['text'].split())}")
    print(f"Text: {chunk['text'][:180]}")
    print("-" * 60)
