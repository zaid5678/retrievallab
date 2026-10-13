
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

documents = [
    "The application stores user information securely and restricts access to authorised personnel.",
    "Our service processes credit card payments and generates invoices.",
    "Employees can reset their passwords through the account settings page.",
    "The database is backed up every night to prevent data loss.",
    "Only approved staff can view confidential customer records.",
]

# A small pretrained model that runs locally.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert every document into a numerical vector.
document_embeddings = model.encode(
    documents,
    convert_to_tensor=True,
    normalize_embeddings=True,
)

#query = "How does the app protect customer data?"
query = "How can we recover information after data is lost?"

query_embedding = model.encode(
    query,
    convert_to_tensor=True,
    normalize_embeddings=True,
)

# Compare the query against every document.
scores = cos_sim(query_embedding, document_embeddings)[0]

# Show the documents from highest to lowest similarity.
ranked_results = sorted(
    zip(scores.tolist(), documents),
    reverse=True,
)

print(f"Query: {query}\n")

for rank, (score, document) in enumerate(ranked_results, start=1):
    print(f"{rank}. Score: {score:.3f}")
    print(f"   {document}\n")
