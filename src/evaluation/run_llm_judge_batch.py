"""
Run LLM-as-Judge evaluation on summary files (offline or merged/online).

Usage:
    # Auto-discover all files (no flags needed)
    python src/evaluation/run_llm_judge_batch.py

    # Single file
    python src/evaluation/run_llm_judge_batch.py \
      --summary-file results/merged-summaries/results-merged-5-20251215_082353.json \
      --limit 3

    # Multiple files
    python src/evaluation/run_llm_judge_batch.py \
      --summary-file results/offline-summaries-JSON-enforced/results-5-offline-0-online-tfidf-20251214_222854.json \
      --summary-file results/merged-summaries/results-merged-5-20251215_082353.json \
      --delay 2

    # Subset by indices
    python src/evaluation/run_llm_judge_batch.py \
      --summary-file results/merged-summaries/results-merged-5-20251215_082353.json \
      --indices 0,2,4

Output: results/evaluation/{offline|merged}/{type}_{k}_llm_judge_scores_{timestamp}.json
"""

import json
import argparse
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from llm_as_judge import (
    llm_score_summary,
    extract_web_docs_from_merged,
    is_error_summary
)


DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
MERGED_CORPUS_DIR = "data/merged-corpus"
OFFLINE_SUMMARIES_DIR = "results/offline-summaries-JSON-enforced"
MERGED_SUMMARIES_DIR = "results/merged-summaries"


def extract_k_from_filename(file_path: str, summary_type: str) -> str:
    """
    Extract k (size) from source filename.
    
    Args:
        file_path: Path to summary file
        summary_type: "merged" or "offline"
        
    Returns:
        k value as string, or "unknown" if pattern doesn't match
    """
    filename = Path(file_path).stem
    
    if summary_type == "merged":
        # Match pattern: results-merged-(\d+)-.*
        match = re.search(r'results-merged-(\d+)', filename, re.IGNORECASE)
        if match:
            return match.group(1)
    elif summary_type == "offline":
        # Match pattern: results-(\d+)-offline-.*
        match = re.search(r'results-(\d+)-offline-', filename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return "unknown"


def generate_output_filename(summary_type: str, k: str, timestamp: Optional[str] = None) -> str:
    """
    Generate output filename in format: {type}_{k}_llm_judge_scores_{timestamp}.json
    
    Args:
        summary_type: "merged" or "offline"
        k: Size value (extracted from source filename)
        timestamp: Optional timestamp string (format: YYYYMMDD_HHMMSS). If None, generates current timestamp.
        
    Returns:
        Filename string
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{summary_type}_{k}_llm_judge_scores_{timestamp}.json"


def get_output_directory(base_output_dir: str, summary_type: str) -> Path:
    """
    Get the appropriate output directory path based on summary type.
    
    Args:
        base_output_dir: Base output directory (e.g., "results/evaluation")
        summary_type: "merged" or "offline"
        
    Returns:
        Path to type-specific subdirectory
    """
    base_path = Path(base_output_dir)
    return base_path / summary_type


def detect_summary_type(file_path: str) -> str:
    """
    Detect if summary is merged/online or offline based on file path.
    
    Args:
        file_path: Path to summary file
        
    Returns:
        "merged" if from merged-summaries, "offline" if from offline-summaries-JSON-enforced
    """
    if "merged-summaries" in file_path:
        return "merged"
    elif "offline-summaries-JSON-enforced" in file_path:
        return "offline"
    else:
        # Default to offline if unclear
        return "offline"


def discover_summary_files() -> List[str]:
    """
    Discover all JSON summary files in the default directories.
    
    Returns:
        List of file paths, sorted: offline files first, then merged files
    """
    files = []
    
    # Discover offline summaries
    offline_dir = Path(OFFLINE_SUMMARIES_DIR)
    if offline_dir.exists():
        offline_files = sorted(offline_dir.glob("*.json"))
        files.extend([str(f) for f in offline_files])
    
    # Discover merged summaries
    merged_dir = Path(MERGED_SUMMARIES_DIR)
    if merged_dir.exists():
        merged_files = sorted(merged_dir.glob("*.json"))
        files.extend([str(f) for f in merged_files])
    
    return files


def find_merged_corpus_file(summary_file_path: str) -> Optional[str]:
    """
    Find corresponding merged corpus file based on summary file.
    
    Args:
        summary_file_path: Path to summary file
        
    Returns:
        Path to merged corpus file, or None if not found
    """
    # Extract number from filename (e.g., merged-5, merged-10, merged-20)
    filename = Path(summary_file_path).stem
    if "merged" in filename.lower():
        # Try to extract number
        match = re.search(r'merged[_-]?(\d+)', filename, re.IGNORECASE)
        if match:
            num = match.group(1)
            corpus_file = Path(MERGED_CORPUS_DIR) / f"merged-{num}.json"
            if corpus_file.exists():
                return str(corpus_file)
    
    # Fallback: try to find any merged corpus file
    corpus_dir = Path(MERGED_CORPUS_DIR)
    if corpus_dir.exists():
        # Try common patterns
        for pattern in ["merged-5.json", "merged-10.json", "merged-20.json"]:
            corpus_file = corpus_dir / pattern
            if corpus_file.exists():
                return str(corpus_file)
    
    return None


def process_summary_file(
    summary_file_path: str,
    limit: Optional[int] = None,
    indices: Optional[List[int]] = None,
    delay: float = 1.0,
    model: str = DEFAULT_MODEL
) -> Dict:
    """
    Process a single summary file and evaluate summaries.
    
    Args:
        summary_file_path: Path to summary file
        limit: Limit number of summaries to evaluate
        indices: Specific indices to evaluate (overrides limit)
        delay: Delay between API calls (seconds)
        model: OpenAI model to use
        
    Returns:
        Dict with evaluation results
    """
    summary_file_path = Path(summary_file_path)
    if not summary_file_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file_path}")
    
    # Detect summary type
    summary_type = detect_summary_type(str(summary_file_path))
    is_merged = (summary_type == "merged")
    
    # Find merged corpus file if needed
    merged_corpus_path = None
    if is_merged:
        merged_corpus_path = find_merged_corpus_file(str(summary_file_path))
        if not merged_corpus_path:
            print(f"Warning: Could not find merged corpus file for {summary_file_path}")
    
    # Load summaries
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    # Apply subset filters
    if indices is not None:
        valid_indices = []
        invalid_indices = []
        for i in indices:
            if 0 <= i < len(summaries):
                valid_indices.append(i)
            else:
                invalid_indices.append(i)
        
        if invalid_indices:
            print(f"Warning: Invalid indices (out of range): {invalid_indices}")
            print(f"Valid range: 0-{len(summaries)-1}")
        
        if not valid_indices:
            raise ValueError(f"No valid indices found. Summary file has {len(summaries)} entries.")
        
        summaries = [summaries[i] for i in valid_indices]
    elif limit is not None:
        if limit <= 0:
            raise ValueError(f"Limit must be positive, got {limit}")
        summaries = summaries[:limit]
    
    print(f"Evaluating {len(summaries)} summaries from {summary_file_path.name} ({summary_type} type)...\n")
    
    results = []
    skipped_errors = 0
    
    for i, entry in enumerate(summaries):
        query = entry.get("query", "")
        entry_id = entry.get("id", None) if isinstance(entry, dict) else None
        
        # Handle summary extraction: prefer "summary" key, fallback to entire entry if it's a dict
        if isinstance(entry, dict) and "summary" in entry:
            summary = entry.get("summary")
        elif isinstance(entry, dict):
            # If entry is a dict but no "summary" key, use the whole entry
            summary = entry
        else:
            # If entry is not a dict, use it directly
            summary = entry
        
        print(f"[{i+1}/{len(summaries)}] {query[:60] if query else 'No query'}...")
        
        # Check if this is an error summary
        if is_error_summary(summary, is_merged=is_merged):
            print(f"    Skipping error summary")
            skipped_errors += 1
            continue
        
        # Extract web docs if merged summary
        web_docs = []
        if is_merged and merged_corpus_path:
            web_docs = extract_web_docs_from_merged(query, merged_corpus_path)
            if web_docs:
                print(f"    Found {len(web_docs)} web docs")
        
        # Evaluate summary
        try:
            scores = llm_score_summary(
                summary=summary,
                query=query,
                web_docs=web_docs if web_docs else None,
                model=model
            )
            
            result_entry = {
                "id": entry_id,
                "query": query,
                "scores": {
                    "criterion_1_claim_relevance": scores.get("criterion_1_claim_relevance", 0),
                    "criterion_2_perspective_claim_alignment": scores.get("criterion_2_perspective_claim_alignment", 0),
                    "criterion_3_perspective_distinctness": scores.get("criterion_3_perspective_distinctness", 0),
                    "criterion_4_coverage_of_core_arguments": scores.get("criterion_4_coverage_of_core_arguments", 0),
                    "criterion_5_factual_grounding": scores.get("criterion_5_factual_grounding", 0),
                    "total_score": scores.get("total_score", 0)
                },
                "error": scores.get("error"),
                "raw_response": scores.get("raw_response", "")
            }
            results.append(result_entry)
            
            print(f"    Score: {scores.get('total_score', 'N/A')}/10")
            if scores.get("error"):
                print(f"    Error: {scores.get('error')}")
            print()
            
        except Exception as e:
            error_msg = str(e)
            print(f"    Error evaluating: {error_msg}\n")
            
            # Try to preserve any partial results if available
            raw_response = ""
            partial_scores = {}
            try:
                # If llm_score_summary was called but failed, we might not have scores
                # But we can at least preserve the query
                pass
            except:
                pass
            
            results.append({
                "id": entry_id,
                "query": query,
                "scores": {
                    "criterion_1_claim_relevance": 0,
                    "criterion_2_perspective_claim_alignment": 0,
                    "criterion_3_perspective_distinctness": 0,
                    "criterion_4_coverage_of_core_arguments": 0,
                    "criterion_5_factual_grounding": 0,
                    "total_score": 0
                },
                "error": error_msg,
                "raw_response": raw_response
            })
        
        # Delay between API calls
        if i < len(summaries) - 1:
            time.sleep(delay)
    
    return {
        "summary_file": str(summary_file_path),
        "summary_type": summary_type,
        "num_evaluated": len(results),
        "num_skipped_errors": skipped_errors,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate summaries with LLM-as-Judge (batch processing)"
    )
    parser.add_argument(
        "--summary-file",
        action="append",
        required=False,
        help="Path to summary file (can be repeated for multiple files). If not provided, automatically discovers all JSON files in results/offline-summaries-JSON-enforced/ and results/merged-summaries/"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of summaries per file"
    )
    parser.add_argument(
        "--indices",
        type=str,
        help="Comma-separated list of indices to evaluate (e.g., '0,2,4')"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation",
        help="Output directory for results (default: results/evaluation). Subdirectories (offline/merged) are created automatically based on summary type."
    )
    
    args = parser.parse_args()
    
    # Parse indices if provided
    indices = None
    if args.indices:
        try:
            indices = [int(i.strip()) for i in args.indices.split(",")]
        except ValueError:
            raise ValueError(f"Invalid indices format: {args.indices}. Use comma-separated integers.")
    
    # Get summary files: use provided files or auto-discover
    if args.summary_file:
        summary_files = args.summary_file
    else:
        summary_files = discover_summary_files()
        if not summary_files:
            print("No summary files found. Please provide --summary-file or ensure files exist in:")
            print(f"  - {OFFLINE_SUMMARIES_DIR}")
            print(f"  - {MERGED_SUMMARIES_DIR}")
            return
        print(f"Auto-discovered {len(summary_files)} summary file(s):")
        for f in summary_files:
            print(f"  - {f}")
        print()
    
    # Process each summary file and save results separately
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_output_files = []
    all_file_results = []
    
    for summary_file in summary_files:
        file_results = process_summary_file(
            summary_file_path=summary_file,
            limit=args.limit,
            indices=indices,
            delay=args.delay,
            model=args.model
        )
        all_file_results.append(file_results)
        
        # Determine output directory and filename based on summary type
        summary_type = file_results["summary_type"]
        output_dir = get_output_directory(str(base_output_dir), summary_type)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract k from source filename
        k = extract_k_from_filename(file_results["summary_file"], summary_type)
        
        # Generate timestamp for filename (format: YYYYMMDD_HHMMSS)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate output filename with timestamp
        output_filename = generate_output_filename(summary_type, k, timestamp_str)
        output_file = output_dir / output_filename
        
        # Prepare output data for this file
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "summary_file": file_results["summary_file"],
            "summary_type": summary_type,
            "num_evaluated": file_results["num_evaluated"],
            "num_skipped_errors": file_results["num_skipped_errors"],
            "results": file_results["results"]
        }
        
        # Save results to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        all_output_files.append(output_file)
    
    # Print summary
    total_evaluated = sum(r["num_evaluated"] for r in all_file_results)
    total_skipped = sum(r["num_skipped_errors"] for r in all_file_results)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}")
    print(f"Files processed: {len(all_output_files)}")
    print(f"Total evaluated: {total_evaluated}")
    print(f"Total skipped (errors): {total_skipped}")
    print(f"\nResults saved to:")
    for output_file in all_output_files:
        print(f"  - {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

