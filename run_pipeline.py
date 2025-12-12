import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from src.utils.io import load_theperspective_dataset
from src.utils.io import load_theperspective_evidence
# from src.retrieval.tfidf_retrieval import retrieve_local_docs
from src.retrieval.web_retrieval import search_web
# from src.validation.entailment import check_entailment
# from src.summarization.merge import merge_documents
# from src.summarization.llm_summary import summarize_query
# from src.evaluation.web_metrics import evaluate_all


def main():
    # Command line arguments (examples):
    #   --dataset theperspective --offline-k 0 --online-k 10 --method tfidf --limit 10
    # Flags:
    #   --dataset   : theperspective | perspectrumx (perspectrumx not yet implemented)
    #   --offline-k : top-k offline (TF-IDF) docs (currently unused in pipeline)
    #   --online-k  : top-k web docs to retrieve via Tavily
    #   --method    : label baked into result filename (e.g., tfidf)
    #   --limit     : truncate dataset for quick tests
    parser = argparse.ArgumentParser(
        description="Web-Augmented Multi-Perspective Summarization Pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=["theperspective", "perspectrumx"],
        required=True,
        default="theperspective",
        help="Dataset to use theperspective or perspectrumx"
    )
    parser.add_argument(
        "--offline-k",
        type=int,
        default=0,
        help="Number of top offline (TF-IDF) documents to retrieve."
    )
    parser.add_argument(
        "--online-k",
        type=int,
        default=5,
        help="Number of top online (web) documents to retrieve."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tfidf",
        help="Retrieval method label to include in filename (e.g., tfidf)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process (e.g., 10 for a quick test)."
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    offline_k = args.offline_k
    online_k = args.online_k
    method = args.method
    limit = args.limit

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / (
        f"results-{offline_k}-offline-{online_k}-online-{method}-{timestamp}.json"
    )

    # Load dataset
    if dataset_name == "theperspective":
        dataset = load_theperspective_dataset("data/theperspective")
    else:
        raise NotImplementedError("Perspectrumx not yet added.")

    total_queries = len(dataset)
    print(f"\nLoaded {total_queries} queries from {dataset_name} dataset.")
    print(f"Using top-{online_k} retrieval for web retrieval.")
    print(f"Saving results to: {output_file}")

    # print(dataset)

    # Load evidence depending on dataset
    # if dataset_name == "theperspective":
    #     evidence = load_theperspective_evidence("data/theperspective")
    # else:
    #     raise NotImplementedError("Perspectrumx not yet added.")

    # Optionally limit dataset for quick tests
    if limit is not None:
        dataset = dataset[:limit]
        print(f"Processing first {len(dataset)} queries due to --limit={limit}.")

    # Go over each query, should be from title section for theperspective
    results = []
    for i, entry in enumerate(dataset):
        query_text = entry["query"]
        print("\n")
        # could remove query: text
        print(f"[{i+1}/{len(dataset)}] Query: {query_text}")
        print("\n")

        # TF-IDF document retrieval
        # local_docs = retrieve_local_docs(query_text, evidence, k=k)
        # print(len(local_docs))
        # print(local_docs)

        # Web retrieval
        web_docs = search_web(query_text, k=online_k)

        # Entailment and novel perspective validation
        # validated_web_docs = []
        # #validated_web_docs = [
        #   check_entailment(doc, entry["perspectives"]) for doc in web_docs
        # ]

        # Merge local documents + web documents
        # merged_corpus = merge_documents(local_docs, validated_web_docs)

        # Summarization
        # summary = summarize_query(query_text, web_docs, entry["claims"])

        #print(f"summary:\n{summary}")

        # Evaluation - calculate metrics for LLM summary compared to gold data
        # metrics = evaluate_all(summary, entry)
        # Store web retrieval output; 'api_k' is included within web_docs
        result_entry = {
            "query": query_text,
            "web_docs": web_docs,
        }
        results.append(result_entry)

        # print(f"Summary metrics: {metrics}\n")

        # Write result to JSON file immediately (streaming to file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Pipeline completed")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()