# Retrieval Module

Implements document retrieval combining local TF-IDF search and web retrieval with Tavily API.

## Components

### `tfidf_retrieval.py`
Local document retrieval using TF-IDF vectorization.

```python
from src.retrieval.tfidf_retrieval import retrieve_local_docs

query = "climate change policy"
local_docs = retrieve_local_docs(query, evidence, k=5)
# Returns: list[dict] with keys: id, content, score
```

### `web_retrieval.py`
Web document retrieval using Tavily API with caching and rate limiting.

```python
from src.retrieval.web_retrieval import search_web

web_docs = search_web("climate change policy", k=3)
# Returns: list[dict] with keys: id, content, url, source_type, title, domain
# Results cached to cache/web_results.json
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

### Caching
- Results are automatically cached to `cache/web_results.json`
- Cache key is based on query and k value
- Clear cache with: `rm cache/web_results.json`

## Features

### Automatic Retry & Rate Limiting
- Exponential backoff on rate limits (1s → 2s → 4s)
- Automatic detection and retry for rate limit errors
- Max 3 retry attempts per request

### Error Handling
- Missing API key: Returns empty list with logged error
- Network errors: Logged and handled gracefully
- Invalid queries: Returns empty list
- Cache errors: Auto-creates cache directory on first write

## Output Format

Both retrieval functions return structured documents:

```python
{
    "id": "unique_hash_identifier",
    "content": "cleaned_document_text",
    "url": "https://source.com",        # web only
    "source_type": "web",
    "title": "Document Title",
    "domain": "source.com"              # web only
}
```

Local retrieval also includes:
- `"score"`: Cosine similarity score (0-1)
