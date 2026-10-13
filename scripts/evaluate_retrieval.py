
import json
from pathlib import Path

from src.retrievallab.retrieval import SemanticRetriever


DATASET_PATH = Path("data/evaluation.json")


def evaluate_configuration(dataset, chunk_size, overlap, k=3):
    retriever = SemanticRetriever(
        chunk_size=chunk_size,
        overlap=overlap,
    )

    ranks = []

    for item in dataset:
        expected = item["expected_evidence"]
        if expected is None:
            continue

        results = retriever.search(item["question"], top_k=k)

        rank = next(
            (
                i
                for i, result in enumerate(results, start=1)
                if expected.casefold() in result["text"].casefold()
            ),
            None,
        )
        ranks.append(rank)

    if not ranks:
        return {"hit_rate_at_1": 0, "recall_at_k": 0, "mrr": 0}

    return {
        "hit_rate_at_1": sum(r == 1 for r in ranks) / len(ranks),
        "recall_at_k": sum(r is not None for r in ranks) / len(ranks),
        "mrr": sum(1 / r if r is not None else 0 for r in ranks) / len(ranks),
    }


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    retriever = SemanticRetriever()
    k = 3

    evidence_ranks = []

    print(f"\nEvidence-level retrieval evaluation (top {k})")
    print("-" * 55)

    for item in dataset:
        expected_evidence = item["expected_evidence"]

        if expected_evidence is None:
            continue

        results = retriever.search(item["question"], top_k=k)

        rank = next(
            (
                index
                for index, result in enumerate(results, start=1)
                if expected_evidence.casefold() in result["text"].casefold()
            ),
            None,
        )

        evidence_ranks.append(rank)

        print(f"\nQuestion: {item['question']}")
        print(f"Expected evidence: {expected_evidence}")
        print(
            "Retrieved chunks:",
            [
                (r["chunk_id"], round(r["score"], 3))
                for r in results
            ],
        )
        print(f"Evidence rank: {rank}")

    evaluated = len(evidence_ranks)
    hits_at_k = sum(
        1 for rank in evidence_ranks if rank is not None
    )
    hits_at_1 = sum(
        1 for rank in evidence_ranks if rank == 1
    )

    recall_at_k = hits_at_k / evaluated if evaluated else 0
    hit_rate_at_1 = hits_at_1 / evaluated if evaluated else 0
    mrr = (
        sum(1 / rank if rank is not None else 0
            for rank in evidence_ranks) / evaluated
        if evaluated
        else 0
    )

    print("\nSummary")
    print("-" * 55)
    print(f"Evaluated questions: {evaluated}")
    print(f"Evidence Recall@{k}: {hits_at_k}/{evaluated} = {recall_at_k:.1%}")
    print(f"Hit Rate@1: {hits_at_1}/{evaluated} = {hit_rate_at_1:.1%}")
    print(f"Evidence MRR@{k}: {mrr:.3f}")


def main2():
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    chunk_sizes = [30, 50, 75, 100]
    overlaps = [0, 10, 20]
    k = 3
    results = []

    print("\nGrid Search: Retrieval Hyperparameters")
    print("-" * 65)

    for size in chunk_sizes:
        for overlap in overlaps:
            # Overlap must be smaller than chunk size.
            if overlap >= size:
                continue

            metrics = evaluate_configuration(
                dataset,
                chunk_size=size,
                overlap=overlap,
                k=k,
            )

            result = {
                "chunk_size": size,
                "overlap": overlap,
                **metrics,
            }
            results.append(result)

            print(
                f"chunk_size={size:3}, "
                f"overlap={overlap:2} | "
                f"Hit@1={metrics['hit_rate_at_1']:.1%} | "
                f"Recall@{k}={metrics['recall_at_k']:.1%} | "
                f"MRR={metrics['mrr']:.3f}"
            )

    # Rank configurations by MRR, then Hit Rate@1,
    # then Recall@k.
    results.sort(
        key=lambda result: (
            result["mrr"],
            result["hit_rate_at_1"],
            result["recall_at_k"],
        ),
        reverse=True,
    )

    print("\nTop 5 configurations")
    print("-" * 65)

    for rank, result in enumerate(results[:5], start=1):
        print(
            f"{rank}. chunk_size={result['chunk_size']}, "
            f"overlap={result['overlap']} | "
            f"Hit@1={result['hit_rate_at_1']:.1%} | "
            f"Recall@{k}={result['recall_at_k']:.1%} | "
            f"MRR={result['mrr']:.3f}"
        )

if __name__ == "__main__":
    main2()
