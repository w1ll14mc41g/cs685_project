# Retrieval Module

Implements document retrieval combining local TF-IDF search and web retrieval with Tavily API.

## Components

### `tfidf_retrieval.py`
Local document retrieval using TF-IDF vectorization and cosine similarity.

**Function**: `retrieve_local_docs(query: str, evidence: list, k: int = 5) -> List[Dict]`

**Input**:
- `query`: Search query string
- `evidence`: List of document dicts with `id` and `content` fields
- `k`: Number of top documents to return

**Output**: List of top-k documents with similarity scores

```python
from src.retrieval.tfidf_retrieval import retrieve_local_docs

query = "climate change policy"
evidence = [{"id": 0, "content": "..."}, ...]
local_docs = retrieve_local_docs(query, evidence, k=5)
# Returns: list[dict] with keys: id (int), content (str), score (float)
```

### `web_retrieval.py`
Web document retrieval using Tavily API with exponential backoff rate limiting.

**Function**: `search_web(query: str, k: int = 3, query_id: str = None, ...) -> Dict`

**Input**:
- `query`: Search query string
- `k`: Desired number of results
- `query_id`: Optional query ID (if provided, returns full entry structure)

**Output**: Dictionary with metadata and results list

```python
from src.retrieval.web_retrieval import search_web

# Basic usage
web_docs = search_web("climate change policy", k=3)
# Returns: {
#   "num_docs": 3,
#   "api_k": 3,          # actual max_results used (may be > k if retries bumped it)
#   "results": [
#       {"id": 0, "content": "...", "url": "...", "source_type": "web", "title": "...", "domain": "..."},
#       ...
#   ]
# }

# With query_id (returns full entry structure matching web files format)
entry = search_web("climate change policy", k=3, query_id="query_0")
# Returns: {
#   "id": "query_0",
#   "query": "climate change policy",
#   "web_docs": {...}
# }
```

## Usage

### Combining Local and Web Results

```python
from src.retrieval.tfidf_retrieval import retrieve_local_docs
from src.retrieval.web_retrieval import search_web

query = "climate change policy"
k_local = 5
k_web = 3

# Get both local and web documents
local_docs = retrieve_local_docs(query, evidence, k=k_local)
web_docs = search_web(query, k=k_web)

# Combine for downstream processing
all_docs = local_docs + web_docs
```

## Configuration

### Tavily API Setup
Set your Tavily API key in `.env`:
```env
TAVILY_API_KEY=tvly-your_actual_api_key
```
Get your free API key from https://tavily.com

## Features

### Automatic Retry & Rate Limiting
- Exponential backoff on rate limits (1s → 2s → 4s)
- Automatic detection and retry for rate limit errors
- Max 3 retry attempts per request
- If fewer than k results are returned, the module increments `api_k` and retries (up to k+10)

### Error Handling
- Missing API key: Returns empty structure with logged error
- Network errors: Logged and handled gracefully
- Rate limit errors: Automatic exponential backoff retry (up to 3 attempts)
- Insufficient results: Automatically increments `api_k` and retries (up to k+10)

### Output Format

- TF-IDF retrieval returns a list of documents:

```python
[
    {
        "id": 1,                    # Integer ID from evidence dataset
        "content": "Document text...",
        "score": 0.87               # Cosine similarity score (0-1)
    },
    ...
]
```

- Web retrieval returns a dictionary with metadata and a list of documents:

```python
{
    "num_docs": 3,          # number of docs returned (trimmed to k)
    "api_k": 4,             # actual Tavily max_results used (may exceed k)
    "results": [
        {
            "id": 0,        # zero-based within the result set
            "content": "cleaned_document_text",
            "url": "https://source.com",
            "source_type": "web",
            "title": "Document Title",
            "domain": "source.com"
        },
        ...
    ]
}
```
