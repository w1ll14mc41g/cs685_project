# Utils Module

Utility functions for dataset loading, metrics computation, and human evaluation context generation.

## Components

### `io.py`
Dataset loading functions for ThePerspective and PerspectrumX datasets.

**Functions**:

#### `load_theperspective_dataset(folder_path: str) -> List[Dict]`
Loads ThePerspective dataset from `data/theperspective/` directory.

**Input Files**:
- `data.jsonl`: Query, claims, and perspectives
- `doc_new.jsonl`: Evidence documents

**Output Format**:
```python
[
    {
        "id": "Entertainment_0",
        "query": "Topic title",
        "claims": ["Claim 1 text", "Claim 2 text"],
        "perspectives": {
            "pro": ["perspective 1", "perspective 2", ...],
            "con": ["perspective 1", "perspective 2", ...]
        },
        "favor_ids": [205, 364],
        "against_ids": [1138, 858]
    },
    ...
]
```

#### `load_theperspective_evidence(folder_path: str) -> List[Dict]`
Loads evidence documents from ThePerspective dataset.

**Output Format**:
```python
[
    {"id": 0, "content": "Document text..."},
    {"id": 1, "content": "Document text..."},
    ...
]
```

**Usage**:
```python
from src.utils.io import load_theperspective_dataset, load_theperspective_evidence

dataset = load_theperspective_dataset("data/theperspective")
evidence = load_theperspective_evidence("data/theperspective")
```

### `compute_dataset_metrics.py`
Computes basic dataset statistics for ThePerspective dataset.

**Usage**:
```bash
python src/utils/compute_dataset_metrics.py
```

**Output**:
- `results/dataset-analysis/dataset-metrics.md`

**Metrics Computed**:
- Basic statistics: number of queries, documents, total words, average words per document
- Data quality: unique queries, duplicate queries
- Task-specific statistics:
  - Average perspectives per query
  - Average words per perspective
  - Average documents per query
  - Average words per claim
- Example input/output pair

**Example Output**:
```markdown
## Dataset Statistics
- **Number of queries:** 185
- **Number of evidence documents:** 4,107
- **Total words across all documents:** 2,345,678
- **Average words per document:** 571.2

## Task-Specific Statistics
- **Average perspectives per query:** 6.0
- **Average words per perspective:** 12.5
- **Average documents per query:** 22.2
```

### `parse_human_judge_context.py`
Generates HTML context files for human evaluation of summaries.

**Usage**:
```bash
python src/utils/parse_human_judge_context.py
```

**Input**:
- `data/valid-queries/summary_eval_{k}.json` - List of valid queries with IDs

**Output**:
- `data/human-eval/llm_as_judge_context-{timestamp}.html`

**Features**:
- Interactive HTML interface with query navigation
- Displays gold reference, offline summary, and merged summary side-by-side
- Shows relevant web documents for merged summaries
- Toggle between offline and merged summary views
- Query selector dropdown for easy navigation

**HTML Structure**:
- Gold reference (from `data/theperspective/data.jsonl`)
- Offline summary (from `results/offline-summaries-JSON-enforced/`)
- Merged summary (from `results/merged-summaries/`)
- Web documents (from `data/merged-corpus/merged-{k}.json`)

**Helper Functions**:
- `get_gold_reference(query: str, gold_file: str) -> Optional[List[Dict]]`: Retrieves gold reference by matching query text
- `get_web_docs(query: str, merged_file: str) -> List[Dict]`: Extracts web docs (URL entries) for a query
- `find_summary(query: str, summaries_file: str) -> Optional[Dict]`: Finds summary entry by matching query text

## Dataset Formats

### ThePerspective Dataset

**`data.jsonl`** (one JSON object per line):
```json
{
    "id": "Entertainment_0",
    "response1": ["perspective 1", "perspective 2"],
    "response2": ["perspective 1", "perspective 2"],
    "favor_ids": [205, 364],
    "against_ids": [1138, 858],
    "t1": "Claim 1 text",
    "t2": "Claim 2 text",
    "title": "Topic title"
}
```

**`doc_new.jsonl`** (one JSON object per line):
```json
{
    "id": 0,
    "content": "Document text..."
}
```

## Related Modules

- **Retrieval** (`src/retrieval/`): Uses dataset loading functions
- **Summarization** (`src/summarization/`): Uses dataset loading functions
- **Evaluation** (`src/evaluation/`): Uses gold references and web docs for evaluation
- **Validation** (`src/validation/`): Uses valid queries for finding evaluation sets

