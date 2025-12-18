# Web-Augmented Multi-Perspective RAG Summarization

A research project that extends the PerSphere framework with web-search grounding to improve multi-perspective summarization quality. This pipeline combines offline TF-IDF retrieval from static corpora with dynamic web retrieval to generate balanced, well-grounded summaries that cover diverse viewpoints on controversial topics.

## Overview

This project implements a 5-stage pipeline for multi-perspective summarization:

1. **Offline Retrieval**: TF-IDF-based retrieval from ThePerspective dataset
2. **Web Retrieval**: Dynamic document retrieval via Tavily API
3. **Relevance Filtering**: GPT-5-Nano LLM-as-Judge for web document relevance classification
4. **Corpus Merging**: Combining offline and validated web documents
5. **Summarization**: Llama-3.2-3B-Instruct generates structured multi-perspective summaries

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for Llama model inference)
- Hugging Face account and access token

### Setup

1. Clone the repository:
```bash
git clone https://github.com/w1ll14mc41g/cs685_project.git
cd cs685_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root:
```env
# Required for LLM-as-Judge evaluation and relevance checking
OPENAI_API_KEY=your_openai_api_key_here

# Required for web retrieval
TAVILY_API_KEY=tvly-your_tavily_api_key_here

# Required for Llama model access
HF_TOKEN=your_huggingface_token_here
```

Get API keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Tavily**: https://tavily.com (free tier: 1,000 searches/month)
- **Hugging Face**: https://huggingface.co/settings/tokens

## Project Structure

```
cs685_project/
├── src/
│   ├── retrieval/          # TF-IDF and web retrieval modules
│   ├── validation/         # Relevance checking with LLM-as-Judge
│   ├── summarization/      # Multi-perspective summary generation
│   ├── evaluation/         # LLM-as-Judge scoring and metrics
│   └── utils/              # Dataset loading and utilities
├── data/
│   ├── theperspective/     # ThePerspective dataset (PerSphere subset)
│   ├── web/                # Cached web retrieval results
│   ├── valid-web/          # Relevance-validated web documents
│   └── merged-corpus/      # Merged offline + web documents
├── results/                # Generated summaries and evaluation scores
├── run_pipeline.py         # Main pipeline entry point
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Generate Summaries

**Offline-only baseline:**
```bash
python run_pipeline.py \
  --dataset theperspective \
  --offline-k 10 \
  --online-k 0 \
  --method tfidf \
  --limit 5
```

**Web-augmented pipeline:**
```bash
python run_pipeline.py \
  --dataset theperspective \
  --offline-k 10 \
  --online-k 10 \
  --method tfidf \
  --limit 5
```

**Using pre-merged corpus:**
```bash
python run_pipeline.py \
  --merged-file data/merged-corpus/merged-10.json \
  --limit 5
```

### 2. Evaluate Summaries

Evaluate summaries using LLM-as-Judge:
```bash
# Auto-discover and evaluate all summary files
python src/evaluation/run_llm_judge_batch.py

# Evaluate specific file
python src/evaluation/run_llm_judge_batch.py \
  --summary-file results/merged-summaries/results-merged-10-20251215_082353.json \
  --limit 10
```

### 3. Compute Web Metrics

Calculate web retrieval metrics (relevance rate, token increase):
```bash
python src/evaluation/web_metrics.py
```

## Key Components

### Retrieval (`src/retrieval/`)

- **TF-IDF Retrieval**: Local document retrieval from ThePerspective dataset
- **Web Retrieval**: Tavily API integration with exponential backoff rate limiting

See `src/retrieval/README.md` for detailed usage.

### Validation (`src/validation/`)

- **Relevance Checker**: GPT-5-Nano classifies web documents as relevant (R) or not relevant (NR)
- **Valid Query Finder**: Identifies queries with valid summaries for evaluation

See `src/validation/README.md` for detailed usage.

### Summarization (`src/summarization/`)

- **LLM Summary**: Generates structured JSON summaries with claims, perspectives, and document citations
- **Merged Summary**: Handles mixed offline (integer IDs) and web (URL string IDs) document formats
- **Merge Logic**: Combines offline and web documents into unified corpus

See `src/summarization/README.md` for detailed usage.

### Evaluation (`src/evaluation/`)

- **LLM-as-Judge**: 5-criteria rubric scoring (0-10 points total):
  - Claim relevance
  - Perspective-claim alignment
  - Perspective distinctness
  - Coverage of core arguments
  - Factual grounding
- **Web Metrics**: Relevance rate, new relevant doc count, token increase rate
- **Local Metrics**: Recall@k, Cover@k for offline retrieval

See `src/evaluation/README.md` for detailed evaluation procedures.

### Utils (`src/utils/`)

- **Dataset Loading**: Functions for loading ThePerspective and PerspectrumX datasets
- **Dataset Metrics**: Computes basic dataset statistics
- **Human Evaluation**: Generates HTML context files for human evaluation

See `src/utils/README.md` for detailed usage.

## Dataset

This project uses **ThePerspective** subset of the PerSphere dataset:
- 185 controversial queries
- 4,107 evidence documents
- Average 6 perspectives per query
- Two opposing claims per query

Dataset files:
- `data/theperspective/data.jsonl` - Query, claims, and perspectives
- `data/theperspective/doc_new.jsonl` - Evidence documents

See `data/README.md` for detailed information about dataset formats, intermediate files, and data flow.

## Results

Key findings from the research:

- **Offline TF-IDF**: Recall@10 = 0.697, Cover@10 = 0.697
- **Web Retrieval**: Relevance rate @ k=10 = 0.944 (94.4% of retrieved docs are relevant)
- **Summary Quality**: Merged summaries score 7.937/10 vs. 7.770/10 for offline-only (k=10)

See `final-report.tex` Section 6 (Results) for complete quantitative analysis with 95% confidence intervals.

See `results/README.md` for detailed information about results directory structure, file formats, and how to interpret evaluation outputs.

## Command-Line Options

### `run_pipeline.py`

- `--dataset`: Dataset to use (`theperspective` or `perspectrumx`)
- `--offline-k`: Number of offline documents to retrieve (default: 0)
- `--online-k`: Number of web documents to retrieve (default: 0)
- `--method`: Retrieval method label (default: `tfidf`)
- `--limit`: Limit number of queries to process (for testing)
- `--merged-file`: Path to pre-merged corpus JSON (bypasses retrieval)

### `run_llm_judge_batch.py`

- `--summary-file`: Path to summary JSON file (can be repeated)
- `--limit`: Limit number of summaries to evaluate
- `--indices`: Evaluate specific indices (comma-separated)
- `--delay`: Delay between API calls in seconds (default: 1.0)
- `--model`: OpenAI model to use (default: `gpt-5-nano-2025-08-07`)
- `--output-dir`: Output directory (default: `results/evaluation`)

## Output Format

### Summary Output

Summaries are saved as JSON with the following structure:
```json
{
  "id": "query_0",
  "query": "Should phones be banned in schools?",
  "summary": [
    {
      "claim": "Schools should ban phones",
      "perspectives": [
        {
          "text": "One-sentence perspective summary",
          "evidence_docs": [1, 5]
        },
        ...
      ]
    },
    {
      "claim": "Schools should allow phones",
      "perspectives": [
        {
          "text": "One-sentence perspective summary",
          "evidence_docs": [2, 8]
        },
        ...
      ]
    }
  ]
}
```

**Note**: For merged summaries, `evidence_docs` may contain mixed types (integers for offline docs, URL strings for web docs). For offline-only summaries, `evidence_docs` contains only integer IDs.

### Evaluation Output

LLM-as-Judge scores include:
- Individual criterion scores (0-2 each)
- Total score (0-10)
- Explanations for each criterion
- Error handling for failed evaluations

## Troubleshooting

**Out of memory errors:**
- Use smaller `k` values (5 or 10 instead of 20)
- Reduce `--limit` for testing
- Ensure GPU has sufficient VRAM for Llama-3.2-3B-Instruct

**API rate limits:**
- Tavily: Free tier allows 1,000 searches/month
- OpenAI: Check your API usage limits
- Use `--delay` flag in evaluation scripts to slow down requests

**JSON parsing errors:**
- The pipeline includes automatic retry logic (up to 10 attempts)
- Check that `outlines` library is properly installed for structured generation

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{persphere2024,
  title={PerSphere: A Multi-Perspective Dataset for Perspective-Aware Summarization},
  author={...},
  journal={...},
  year={2024}
}
```

## License

See repository for license information.

## Contributors

- Aryan Sajith (asajith@umass.edu)
- Steven Ren (skren@umass.edu)
- William Cai (wlcai@umass.edu)

## Acknowledgments

This project extends the PerSphere framework and uses ThePerspective dataset. We thank the PerSphere authors for their foundational work on multi-perspective summarization.
