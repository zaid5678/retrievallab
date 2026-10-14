import json
import time
from pathlib import Path

from src.retrievallab.evaluation import first_relevant_rank, retrieval_metrics
from src.retrievallab.nist import control_chunks, load_controls, word_chunks
from src.retrievallab.retrieval import SemanticRetriever


DATASET_PATH = Path("data/eval/nist_questionnaire.json")
RESULTS_PATH = Path("results/nist_eval.json")
K = 5


def evaluate(name: str, chunks: list[dict], dataset: list[dict]) -> dict:
    start = time.perf_counter()
    retriever = SemanticRetriever(chunks=chunks)
    index_seconds = time.perf_counter() - start

    ranks = []
    base_ranks = []
    misses = []
    query_start = time.perf_counter()

    for item in dataset:
        results = retriever.search(item["question"], top_k=K)
        rank = first_relevant_rank(results, item["expected_controls"])
        ranks.append(rank)
        base_ranks.append(
            first_relevant_rank(results, item["expected_controls"], match_base=True)
        )

        if rank is None:
            misses.append(
                {
                    "question": item["question"],
                    "expected": item["expected_controls"],
                    "retrieved": [r["chunk_id"] for r in results],
                }
            )

    query_ms = (time.perf_counter() - query_start) * 1000 / len(dataset)

    return {
        "config": name,
        "chunks": len(chunks),
        "controls_per_chunk": round(
            sum(len(chunk["control_ids"]) for chunk in chunks) / len(chunks), 2
        ),
        **retrieval_metrics(ranks, K),
        f"base_recall_at_{K}": retrieval_metrics(base_ranks, K)[f"recall_at_{K}"],
        "index_seconds": round(index_seconds, 1),
        "query_ms": round(query_ms, 1),
        "misses": misses,
    }


def main():
    controls = load_controls()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    answerable = [item for item in dataset if item["expected_controls"]]

    configs = {
        "words-100/20": word_chunks(controls, chunk_size=100, overlap=20),
        "words-200/40": word_chunks(controls, chunk_size=200, overlap=40),
        "per-control": control_chunks(controls, include_guidance=True),
        "per-control (no discussion)": control_chunks(
            controls, include_guidance=False
        ),
    }

    print(
        f"\nNIST SP 800-53 retrieval: {len(controls)} controls, "
        f"{len(answerable)} answerable questions, k={K}\n"
    )
    header = (
        f"{'config':<30}{'chunks':>8}{'ctl/chk':>9}{'Hit@1':>8}"
        f"{f'R@{K}':>8}{'MRR':>8}{f'baseR@{K}':>10}{'ms/q':>8}"
    )
    print(header)
    print("-" * len(header))

    reports = []
    for name, chunks in configs.items():
        report = evaluate(name, chunks, answerable)
        reports.append(report)
        print(
            f"{name:<30}{report['chunks']:>8}{report['controls_per_chunk']:>9.2f}"
            f"{report['hit_rate_at_1']:>8.3f}{report[f'recall_at_{K}']:>8.3f}"
            f"{report['mrr']:>8.3f}{report[f'base_recall_at_{K}']:>10.3f}"
            f"{report['query_ms']:>8.1f}"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nFull results, including misses, written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
