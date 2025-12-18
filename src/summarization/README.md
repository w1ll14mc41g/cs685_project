# Summarization Module

Generates multi-perspective summaries using Llama-3.2-3B-Instruct with constrained JSON decoding. Supports both offline-only documents (integer IDs) and merged corpus (mixed int/string IDs including URLs).

## Components

### `llm_summary.py`
Generates summaries for offline-only documents with integer document IDs.

**Function**: `summarize_query(query: str, merged_corpus: list) -> List[Dict]`

**Input**:
- `query`: The query/topic
- `merged_corpus`: List of documents with `id` (int), `content`, and optional `score`

**Output**:
```python
[
    {
        "claim": "Positive claim text",
        "perspectives": [
            {
                "text": "One-sentence perspective summary",
                "evidence_docs": [1, 5]  # List of integer document IDs
            },
            ...
        ]
    },
    {
        "claim": "Negative claim text",
        "perspectives": [...]
    }
]
```

**Features**:
- Uses Pydantic models for schema validation
- Constrained JSON generation with `outlines` library
- Automatic retry logic (up to 10 attempts)
- Error handling with fallback summaries

### `llm_summary_merged.py`
Generates summaries for merged corpus with mixed document ID types (integers and URL strings).

**Function**: `summarize_query(query: str, merged_corpus: list) -> List[Dict]`

**Input**:
- `query`: The query/topic
- `merged_corpus`: List of documents with `id` (int or string/URL), `content`, and optional `score`

**Output**:
```python
[
    {
        "claim": "Positive claim text",
        "perspectives": [
            {
                "text": "One-sentence perspective summary",
                "evidence_docs": [1, "https://example.com/doc"]  # Mixed int/string IDs
            },
            ...
        ]
    },
    ...
]
```

**Key Differences from `llm_summary.py`**:
- Accepts mixed document ID types (int and string/URL)
- Normalizes evidence IDs (strips "Doc " prefixes, handles numeric strings)
- Early exit on repeated JSON parse errors (5+ errors indicates prompt too long)
- Specialized error messages for merged corpus context

### `merge.py`
Merges offline and web documents into unified corpus.

**Functions**:

#### `merge_docs_lists(local_docs, web_docs) -> List[Dict]`
Merges local documents with relevant web documents.

**Input**:
- `local_docs`: List of offline documents with integer IDs
- `web_docs`: List of web documents with `relevance` field ("R" or "NR")

**Output**:
- Combined list with:
  - Local docs: `{"id": int, "content": str, "score": float}`
  - Relevant web docs: `{"id": str (URL), "content": str}`

**Filtering**:
- Only includes web documents with `relevance == "R"`
- Transforms web docs from `{"url": "...", "content": "..."}` to `{"id": url, "content": "..."}`

## JSON Schema (Pydantic Models)

### `Perspective`
```python
class Perspective(BaseModel):
    text: str  # One-sentence perspective summary (non-empty)
    evidence_docs: List[Union[int, str]]  # Document IDs (at least one required)
```

### `Claim`
```python
class Claim(BaseModel):
    claim: str  # The claim in response to the query (non-empty)
    perspectives: List[Perspective]  # List of perspectives supporting this claim
```

### `MultiPerspectiveSummary`
```python
class MultiPerspectiveSummary(BaseModel):
    summaries: List[Claim]  # Exactly 2 claims (min_items=2, max_items=2)
```

## Usage

### Offline-Only Summarization
```python
from src.summarization.llm_summary import summarize_query

query = "Should phones be banned in schools?"
offline_docs = [
    {"id": 1, "content": "Document 1 text...", "score": 0.87},
    {"id": 2, "content": "Document 2 text...", "score": 0.82}
]

summary = summarize_query(query, offline_docs)
```

### Merged Corpus Summarization
```python
from src.summarization.llm_summary_merged import summarize_query
from src.summarization.merge import merge_docs_lists

query = "Should phones be banned in schools?"
local_docs = [{"id": 1, "content": "...", "score": 0.87}]
web_docs = [
    {"url": "https://example.com", "content": "...", "relevance": "R"},
    {"url": "https://example2.com", "content": "...", "relevance": "NR"}
]

merged_corpus = merge_docs_lists(local_docs, web_docs)
summary = summarize_query(query, merged_corpus)
```

## Requirements

**Hardware**:
- CUDA-capable GPU (required for Llama-3.2-3B-Instruct inference)

**Environment Variables**:
- `HF_TOKEN`: Hugging Face access token (required for model download)

**Python Dependencies**:
- `transformers` - Hugging Face model loading
- `torch` - PyTorch for model inference
- `outlines` - Constrained JSON generation
- `pydantic` - Schema validation

## Model Details

**Model**: `meta-llama/Llama-3.2-3B-Instruct`

**Generation Parameters**:
- `max_new_tokens`: 1500
- `temperature`: 0.1
- `top_p`: 0.8

**Model Caching**:
- Models are cached at module level to avoid reloading
- First call loads model; subsequent calls reuse cached model

## Error Handling

### Retry Logic
- **Offline summaries**: Up to 10 retry attempts
- **Merged summaries**: Up to 10 retry attempts, with early exit after 5+ JSON parse errors

### Fallback Behavior
When all retries fail, returns fallback summary with error message:
```python
[
    {
        "claim": "Positive claim",
        "perspectives": [{
            "text": "Error generating summary: <error_message>",
            "evidence_docs": [<first_3_doc_ids>]
        }]
    },
    {
        "claim": "Negative claim",
        "perspectives": [{
            "text": "Error generating summary: <error_message>",
            "evidence_docs": [<first_3_doc_ids>]
        }]
    }
]
```

### Error Detection
The validation module (`src/validation/find_valid_queries.py`) detects these error patterns:
- Merged: "JSON parse errors 5+ times (prompt too long)" or "All 10 generation attempts failed"
- Offline: Perspective text starting with "Error generating summary:"

## Prompt Structure

The summarization prompt includes:
1. Query text
2. Document corpus (formatted as `[Doc {id}]: {content}`)
3. Rules for generation:
   - Ignore off-topic documents
   - Include positive and negative claims
   - One-sentence perspectives (max 20-30 words)
   - Each perspective must reference document IDs
   - Each document ID used only once
   - Prefer multiple distinct documents per claim

## Related Modules

- **Retrieval** (`src/retrieval/`): Provides documents for summarization
- **Validation** (`src/validation/`): Validates web documents before merging
- **Evaluation** (`src/evaluation/`): Evaluates generated summaries

