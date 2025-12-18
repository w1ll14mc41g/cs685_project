# Validation Module

This module provides tools for validating web documents and identifying valid queries for evaluation. It uses GPT-5-Nano (LLM-as-Judge) to classify web documents as relevant or not relevant to queries.

## Components

### `relevance_checker.py`
Core relevance classification function using GPT-5-Nano.

**Function**: `check_relevance(query: str, web_docs: list) -> dict`

Classifies web documents as "R" (Relevant) or "NR" (Not Relevant) to a given query.

**Input**:
- `query`: The search query/question
- `web_docs`: List of document dicts with `id` and `content` fields

**Output**:
- Dictionary mapping document IDs (as strings) to "R" or "NR"
- Example: `{"0": "R", "1": "NR", "2": "R"}`

**Features**:
- Uses OpenAI's JSON mode for structured output
- Automatic retry logic (up to 3 attempts)
- Validates and fills missing IDs with "NR" default
- Handles JSON parsing errors gracefully

### `run_relevance_check.py`
Batch processor for classifying web documents across multiple queries.

**Usage**:
```bash
# Process a single file
python src/validation/run_relevance_check.py --input web-5.json --limit 3

# Process all web-*.json files
python src/validation/run_relevance_check.py

# Dry run (test without saving)
python src/validation/run_relevance_check.py --input web-5.json --limit 3 --dry-run
```

**Command-Line Options**:
- `--input`: Specific file to process (e.g., `web-5.json`). If omitted, processes all `web-*.json` files
- `--limit N`: Process only first N queries (useful for testing)
- `--dry-run`: Process and print selected queries without saving

**Input Files**:
- `data/web/web-{k}.json` - Web retrieval results (k = 5, 10, 20)

**Output Files**:
- `data/valid-web/valid-web-{k}.json` - Web documents with relevance labels

**Output Format**:
Each document in the output includes a `relevance` field:
```json
{
  "id": "query_0",
  "query": "Should phones be banned in schools?",
  "web_docs": {
    "results": [
      {
        "id": 0,
        "content": "...",
        "url": "https://...",
        "relevance": "R"
      }
    ]
  }
}
```

### `find_valid_queries.py`
Identifies queries that have valid (non-error) summaries in both offline and merged summary files.

**Usage**:
```bash
python src/validation/find_valid_queries.py
```

**Function**: `find_intersection(offline_file: str, merged_file: str, deduplicate: bool = True) -> List[Dict]`

Finds queries that are valid in BOTH offline and merged files by:
1. Loading summaries from both files
2. Detecting error patterns in each summary
3. Finding intersection of valid queries
4. Optionally deduplicating by query text

**Error Detection Patterns**:
- **Merged summaries**: 
  - "JSON parse errors 5+ times (prompt too long)"
  - "All 10 generation attempts failed"
- **Offline summaries**:
  - Perspective text starting with "Error generating summary:"

**Output**:
- Saves to `data/valid-queries/valid-k-{k}-queries-{timestamp}.json`
- Format: List of dicts with `id_offline`, `id_merged`, and `query` fields

## Setup

**Requirements**:
- `OPENAI_API_KEY` in `.env` file
- OpenAI model supporting JSON mode (e.g., `gpt-5-nano-2025-08-07`)

## Workflow

1. **Web Retrieval** → `data/web/web-{k}.json`
2. **Relevance Checking** → `data/valid-web/valid-web-{k}.json`
3. **Corpus Merging** → `data/merged-corpus/merged-{k}.json`
4. **Summary Generation** → `results/merged-summaries/` and `results/offline-summaries-JSON-enforced/`
5. **Valid Query Finding** → `data/valid-queries/valid-k-{k}-queries-{timestamp}.json`

## Related Modules

- **Retrieval** (`src/retrieval/`): Generates web documents for validation
- **Summarization** (`src/summarization/`): Generates summaries that are validated
- **Evaluation** (`src/evaluation/`): Uses valid queries for LLM-as-Judge evaluation

