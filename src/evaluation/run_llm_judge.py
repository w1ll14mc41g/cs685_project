"""
Run LLM-as-Judge evaluation on offline summaries.

Usage:
    python -m src.evaluation.run_llm_judge --limit 3   # Test on 3 summaries
    python -m src.evaluation.run_llm_judge             # Run on all

Output: results/evaluation/llm_judge_scores.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from llm_as_judge import llm_score_summary

INPUT_PATH = "data/offline/offline-summaries.json"
OUTPUT_PATH = "results/evaluation/llm_judge_scores.json"
DEFAULT_MODEL = "gpt-5-nano-2025-08-07"

def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries with LLM-as-Judge")
    parser.add_argument("--limit", type=int, help="Limit number of summaries")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model")
    args = parser.parse_args()

    # Load summaries
    with open(INPUT_PATH) as f:
        data = json.load(f)
    
    if args.limit:
        data = data[:args.limit]
    
    print(f"Evaluating {len(data)} summaries with {args.model}...\n")
    
    results = []
    for i, entry in enumerate(data):
        query = entry["query"]
        print(f"[{i+1}/{len(data)}] {query[:50]}...")
        
        scores = llm_score_summary(entry["summary"], model=args.model)
        
        results.append({
            "query": query,
            "scores": {k: scores.get(k) for k in [
                "claim_perspective_alignment", "evidence_support",
                "perspective_distinctiveness", "query_coverage",
                "content_groundedness", "total_score"
            ]},
            "error": scores.get("error"),
            "raw_response": scores.get("raw_response", "")
        })
        
        print(f"    Score: {scores.get('total_score', 'N/A')}/10\n")
    
    # Save
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "num_evaluated": len(results),
            "results": results
        }, f, indent=2)
    
    print(f"Saved to: {output_path}")
    
if __name__ == "__main__":
    main()
