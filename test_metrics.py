import json
import csv
from pathlib import Path
from datetime import datetime
import statistics
import math
import numpy as np

from src.utils.io import load_theperspective_dataset, load_theperspective_evidence
from src.evaluation.local_metrics import recall_at_k, cover_at_k
from src.retrieval.tfidf_retrieval import retrieve_local_docs

def bootstrap_ci(data, n_boot=1000, ci=95):
    if len(data) == 0:
        return np.nan, np.nan
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


def quartiles(xs):
    if not xs:
        return (math.nan, math.nan, math.nan, math.nan)
    xs = sorted(xs)
    q1 = xs[int(0.25 * (len(xs) - 1))]
    med = statistics.median(xs)
    q3 = xs[int(0.75 * (len(xs) - 1))]
    q4 = xs[-1]
    return q1, med, q3, q4

def main(ks=(5, 10, 20)):
    out_dir = Path("results/")  # store results results folder

    # Load dataset and evidence directly
    data = load_theperspective_dataset("data/theperspective")
    evidence = load_theperspective_evidence("data/theperspective")

    per_query_rows = []
    aggregate_results = {}

    # Loop over k
    for k in ks:
        recall_vals = []
        cover_vals = []

        for doc in data:
            query = doc.get("query")
            if not query:
                continue

            # TF-IDF retrieval
            top_docs = retrieve_local_docs(query, evidence, k=k)
            retrieved_ids = [d.get("id") for d in top_docs]

            # Gold evidence
            gold_ids = doc.get("favor_ids", []) + doc.get("against_ids", [])
            if not gold_ids:
                continue

            # Metrics
            r = recall_at_k(retrieved_ids, gold_ids, k=k)
            c = cover_at_k(retrieved_ids, gold_ids)

            recall_vals.append(r)
            cover_vals.append(c)

            per_query_rows.append({
                "query": query,
                "k": k,
                "recall_at_k": r,
                "cover_at_k": c,
                "retrieved_ids": json.dumps(retrieved_ids),
                "gold_ids": json.dumps(gold_ids),
            })

        # Aggregate statistics
        recall_q1, recall_med, recall_q3, recall_q4 = quartiles(recall_vals)
        cover_q1, cover_med, cover_q3, cover_q4 = quartiles(cover_vals)
        recall_ci_lo, recall_ci_hi = bootstrap_ci(recall_vals)
        cover_ci_lo, cover_ci_hi = bootstrap_ci(cover_vals)

        aggregate_results[k] = {
            "recall_at_k": {
                "mean": statistics.mean(recall_vals) if recall_vals else 0.0,
                "median": recall_med,
                "q1": recall_q1,
                "q3": recall_q3,
                "q4": recall_q4,
                "ci_95_lower": recall_ci_lo,
                "ci_95_upper": recall_ci_hi,
                "n": len(recall_vals),
            },
            "cover_at_k": {
                "mean": statistics.mean(cover_vals) if cover_vals else 0.0,
                "median": cover_med,
                "q1": cover_q1,
                "q3": cover_q3,
                "q4": cover_q4,
                "ci_95_lower": cover_ci_lo,
                "ci_95_upper": cover_ci_hi,
                "n": len(cover_vals),
            },
        }

        print(f"\nk={k}")
        print(f"  recall@{k} mean: {aggregate_results[k]['recall_at_k']['mean']:.4f}")
        print(f"  cover@{k}  mean: {aggregate_results[k]['cover_at_k']['mean']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-query CSV
    per_query_csv = out_dir / f"tfidf_per_query_metrics_{timestamp}.csv"
    with open(per_query_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "k", "recall_at_k", "cover_at_k", "retrieved_ids", "gold_ids"],
        )
        writer.writeheader()
        writer.writerows(per_query_rows)

    # Aggregate JSON
    aggregate_json = out_dir / f"tfidf_aggregate_metrics_{timestamp}.json"
    with open(aggregate_json, "w", encoding="utf-8") as f:
        json.dump(aggregate_results, f, indent=2)

    print("\nResults stored in current directory:")
    print(f"  Per-query CSV: {per_query_csv.name}")
    print(f"  Aggregate JSON: {aggregate_json.name}")


if __name__ == "__main__":
    main()
