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
Web document retrieval using Tavily API with aggressive caching and rate limiting.

```python
from src.retrieval.web_retrieval import search_web

web_docs = search_web("climate change policy", k=3)
# Returns: list[dict] with keys: id, content, url, source_type, title, domain
# Results cached to cache/web_results.json
```

## Additive Retrieval Strategy

Combine local and web results for each query:

```python
from src.retrieval.tfidf_retrieval import retrieve_local_docs
from src.retrieval.web_retrieval import search_web

query = "climate change policy"
k_local = 5
k_web = 3

# Get both local and web documents
local_docs = retrieve_local_docs(query, evidence, k=k_local)
web_docs = search_web(query, k=k_web)

# Combine for next stage (entailment filtering)
all_docs = local_docs + web_docs
```

## Web Retrieval Details

### Caching Strategy (Cost: $0)
- **First call**: Makes Tavily API request (counts against 1,000/month quota)
- **Second call** with same query+k: Returns from cache instantly
- **Cache key**: MD5 hash of `f"{query}:{k}"` → deterministic & query-specific
- **Cache location**: `cache/web_results.json`
- **Cache size**: Expected 3-6 MB for 555 documents from 185 queries
- **Clear cache**: `rm cache/web_results.json`

**Why different k matters**: `search_web("query", k=3)` and `search_web("query", k=5)` use different cache entries since their hash keys differ.

### Rate Limiting
- **Max retries**: 3 attempts
- **Backoff schedule**: 1s → 2s → 4s (exponential backoff)
- **Detection**: Automatically detects rate limit errors (429, "rate_limit" in message)

### Configuration
Set Tavily API key in `.env`:
```env
TAVILY_API_KEY=tvly-your_actual_api_key
```
Get key from https://tavily.com (free tier: 1,000 searches/month)

### Cost Estimation
| Component | Count | Cost |
|-----------|-------|------|
| Tavily searches | 185 queries | $0 |
| Usage rate | 18.5% of quota | Safe |
| LLM entailment | 600-700 (filtered) | $0 (local Llama) |
| **Total** | - | **$0** |

### Error Handling
| Scenario | Behavior |
|----------|----------|
| Missing TAVILY_API_KEY | Returns empty list, logs error |
| Rate limit (429) | Retries with exponential backoff |
| Network error | Logs error, returns empty list |
| Invalid query | Returns empty list (Tavily handles) |
| Cache errors | Auto-creates cache dir on first write |

## Output Format

Both retrieval functions return structured documents:

```python
{
    "id": "unique_hash_identifier",
    "content": "cleaned_document_text",
    "url": "https://source.com",  # web only
    "source_type": "web",
    "title": "Document Title",
    "domain": "source.com"
}
```

Local retrieval also includes:
- `"score"`: Cosine similarity score (0-1)

## Monitoring & Logging

All API calls logged with INFO level:
```
INFO:src.retrieval.web_retrieval:Cache hit for query: climate change...
INFO:src.retrieval.web_retrieval:Tavily API call for query: policy (attempt 1/3)
INFO:src.retrieval.web_retrieval:Retrieved 3 web documents
WARNING:src.retrieval.web_retrieval:Rate limited. Backing off for 2.0s...
```

Track usage:
```bash
# Check cache size
ls -lh cache/web_results.json

# Count cached queries
jq 'keys | length' cache/web_results.json

# Monitor logs for API calls
grep "API call" output.log
```

## Next Steps

**Day 3-4**: Implement entailment filtering in `src/validation/entailment.py`
- Similarity pre-filter (sentence-transformers, local)
- LLM entailment check (Llama-3.1-8B)
- Reduces LLM calls from 1,665 to 600-700

See main project plan for complete timeline.
