# LLM-as-Judge Evaluation System

Evaluates multi-perspective summaries using a 5-criteria rubric (10 points total) that works for both offline-only and web-augmented summaries.

## Scoring Criteria

Each summary is evaluated on 5 criteria (0-2 points each, total 0-10):

1. **Claim Relevance** (0-2) - Do claims address the query and are they oppositional?
2. **Perspective-Claim Alignment** (0-2) - Do perspectives support their claims?
3. **Perspective Distinctness** (0-2) - Are perspectives non-overlapping?
4. **Coverage of Core Arguments** (0-2) - Does it cover key arguments from evidence?
5. **Factual Grounding** (0-2) - Are perspectives supported by evidence?

The model assigns individual criterion scores; the total score is computed automatically by summing them.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages: `pydantic>=2.0.0`, `openai>=1.0.0`, `python-dotenv>=1.0.0`

2. Set up API key:
   - Create a `.env` file in the project root directory
   - Add: `OPENAI_API_KEY=your_api_key_here`
   - Requires OpenAI model supporting Structured Outputs (e.g., `gpt-5-nano-2025-08-07`, `gpt-4o-2024-08-06`)

## Usage

### Auto-Discovery (Recommended for Batch Jobs)

When no `--summary-file` is provided, the script automatically discovers and processes all JSON files in:
- `results/offline-summaries-JSON-enforced/` (processed first)
- `results/merged-summaries/` (processed second)

```bash
# Process all files automatically
python src/evaluation/run_llm_judge_batch.py
```

### Manual File Selection

```bash
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
```

### Command-Line Options

- `--summary-file`: Path to summary file (can be repeated for multiple files). If omitted, auto-discovers all JSON files in default directories.
- `--limit N`: Limit number of summaries to evaluate per file
- `--indices 0,2,4`: Evaluate specific indices (comma-separated)
- `--delay SECONDS`: Delay between API calls (default: 1.0)
- `--model MODEL`: OpenAI model to use (default: gpt-5-nano-2025-08-07)
- `--output-dir DIR`: Output directory (default: results/evaluation)

## Output Format

Results are saved to type-specific subdirectories with timestamps:
- `results/evaluation/offline/offline_{k}_llm_judge_scores_{timestamp}.json`
- `results/evaluation/merged/merged_{k}_llm_judge_scores_{timestamp}.json`

Example: `results/evaluation/merged/merged_20_llm_judge_scores_20251215_143022.json`

Output structure:
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "model": "gpt-5-nano-2025-08-07",
  "summary_file": "results/merged-summaries/results-merged-5-20251215_082353.json",
  "summary_type": "merged",
  "num_evaluated": 3,
  "num_skipped_errors": 0,
  "results": [
    {
      "id": "query_0",
      "query": "...",
      "scores": {
        "criterion_1_claim_relevance": 2,
        "criterion_2_perspective_claim_alignment": 2,
        "criterion_3_perspective_distinctness": 2,
        "criterion_4_coverage_of_core_arguments": 2,
        "criterion_5_factual_grounding": 2,
        "total_score": 10
      },
      "error": null,
      "raw_response": "..."
    }
  ]
}
```

## Implementation Details

### Structured Outputs with Pydantic

Uses OpenAI's **Responses API** with **Structured Outputs** via **Pydantic models**:
- **Schema Adherence**: Guaranteed response format matching the defined schema
- **Type Safety**: Pydantic validates all responses at runtime
- **No Parsing Errors**: Eliminates need for JSON parsing, retries, or fallback logic

Pydantic models:
- `Scores`: Individual criterion scores (0-2 each)
- `Explanations`: Text explanations for each criterion
- `EvaluationResponse`: Complete response containing both scores and explanations

### Error Handling

1. **Pre-Evaluation Error Detection**: Automatically skips summaries with error patterns:
   - Merged summaries: "JSON parse errors 5+ times (prompt too long)" or "All 10 generation attempts failed"
   - Offline summaries: Perspective text starting with "Error generating summary:"
   - Counted in `num_skipped_errors` field

2. **Model Refusals**: Detects and handles safety-based refusals, returning error format with refusal reason.

3. **API Errors**: Catches network issues, rate limits, etc., returning error message with all scores set to 0.

Error responses include `error` field, `raw_response` for debugging, and all score fields set to 0.
