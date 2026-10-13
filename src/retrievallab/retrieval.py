
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.retrievallab.chunking import load_documents, chunk_documents


class SemanticRetriever:
    def __init__(
        self,
        chunk_size: int = 50,
        overlap: int = 10,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

        documents = load_documents()
        self.chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not self.chunks:
            raise ValueError("No document chunks found.")

        self.embeddings = self.model.encode(
            [chunk["text"] for chunk in self.chunks],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = cos_sim(query_embedding, self.embeddings)[0]
        ranked_indices = scores.argsort(descending=True).tolist()

        results = []
        for index in ranked_indices[:top_k]:
            results.append({
                **self.chunks[index],
                "score": float(scores[index]),
            })

        return results
