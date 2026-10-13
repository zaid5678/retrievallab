# RetrievalLab 🔎

**An evaluation-driven Retrieval-Augmented Generation (RAG) search engine built with Python and local LLMs.**

RetrievalLab explores how to build, evaluate, and improve a RAG pipeline from the ground up. It combines semantic search using Sentence Transformers with locally hosted language models through Ollama to answer questions using retrieved document evidence.

The goal is not just to generate answers, but to understand **whether the right evidence is retrieved and how retrieval choices affect answer quality**.

## ✨ Features

- **Document ingestion:** Load Markdown and text documents with source metadata.
- **Configurable chunking:** Split documents into overlapping, word-based chunks.
- **Semantic search:** Generate embeddings with `all-MiniLM-L6-v2` and rank chunks using cosine similarity.
- **Local LLM generation:** Generate evidence-grounded answers using Ollama and `qwen2.5:3b`.
- **Source-aware answers:** Preserve filenames so retrieved evidence can be traced to its source.
- **Retrieval evaluation:** Measure Evidence Recall@k, Hit Rate@1, and Mean Reciprocal Rank (MRR).
- **Hyperparameter experiments:** Compare chunk sizes and overlap settings using grid search.
- **Unit tests:** Test chunking behavior and prompt construction.

## 🧠 How It Works

```text
Markdown / Text Documents
          |
          v
   Document Chunking
          |
          v
   Text Embeddings
  (Sentence Transformers)
          |
          v
  Cosine Similarity Search
          |
          v
   Top-k Evidence Chunks
          |
          v
    Local LLM (Ollama)
          |
          v
  Answer with Source References
```

RetrievalLab separates retrieval from generation so each stage can be inspected and evaluated independently.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Similarity | Cosine similarity |
| Local LLM runtime | Ollama |
| Generation model | `qwen2.5:3b` |
| Testing | pytest |

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or newer
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/retrievallab.git
cd retrievallab
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
python -m pip install -e .
python -m pip install ollama pytest
```

### 4. Download the local language model

```bash
ollama pull qwen2.5:3b
```

Make sure the Ollama service is running before generating answers.

The embedding model is downloaded automatically by Sentence Transformers the first time it is initialized.

### 5. Run a semantic search

```bash
python scripts/search_documents.py
```

Follow the script's prompts or options, if provided, to search your documents.

### 6. Ask a question using RAG

```bash
python scripts/ask_rag.py
```

The pipeline retrieves relevant chunks and passes them to the local LLM to generate an answer grounded in the supplied evidence.

### 7. Evaluate retrieval quality

```bash
python scripts/evaluate_retrieval.py
```

This evaluates retrieval against the question-and-evidence pairs in `data/evaluation.json`.

### 8. Run the tests

```bash
python -m pytest -q
```

## 📊 Evaluation

RetrievalLab evaluates retrieval using three metrics:

- **Evidence Recall@k:** The proportion of answerable questions whose expected evidence appears within the top-k retrieved chunks.
- **Hit Rate@1:** The proportion of answerable questions whose expected evidence appears in the first retrieved chunk.
- **Mean Reciprocal Rank (MRR):** Rewards configurations that rank the expected evidence higher.

The current evaluation dataset is intentionally small and intended for experimentation, not as a general benchmark.

> Evaluation results depend on the dataset, chunking strategy, embedding model, and evidence-matching method. Scores on a small dataset should not be interpreted as proof of production-level retrieval quality.

## 🧪 Experiments

The project explores how retrieval performance changes with parameters such as:

- Chunk size
- Chunk overlap
- Number of retrieved chunks (`top_k`)

A grid search can compare configurations using consistent evaluation questions and ranking metrics.

Future experiments may include hybrid keyword and semantic retrieval, reranking, and answer faithfulness evaluation.

## 📁 Project Structure

```text
retrievallab/
├── data/
│   ├── documents/
│   │   ├── security.md
│   │   └── backups.md
│   └── evaluation.json
├── scripts/
│   ├── ask_rag.py
│   ├── evaluate_retrieval.py
│   ├── inspect_chunks.py
│   └── search_documents.py
├── src/
│   └── retrievallab/
│       ├── __init__.py
│       ├── chunking.py
│       ├── rag.py
│       └── retrieval.py
├── tests/
├── pyproject.toml
└── README.md
```

## 🗺️ Roadmap

- [x] Markdown and text document loading
- [x] Overlapping document chunking
- [x] Embedding-based semantic retrieval
- [x] Local LLM answer generation
- [x] Retrieval metrics and evaluation dataset
- [x] Chunk parameter grid search
- [ ] Hybrid search with BM25 and dense retrieval
- [ ] Cross-encoder reranking
- [ ] Answer faithfulness and citation evaluation
- [ ] Larger, more diverse evaluation dataset
- [ ] Retrieval and generation latency measurements

## 🎯 Project Goals

RetrievalLab is a hands-on exploration of modern retrieval systems, with an emphasis on reproducible experiments, measurable improvements, and understanding the engineering trade-offs behind RAG.

## 📄 License

Consider adding an MIT license if you want others to freely use, modify, and distribute the project. Add a `LICENSE` file before claiming the repository is MIT-licensed.
