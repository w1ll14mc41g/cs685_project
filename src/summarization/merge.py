import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import os


def _call_model_openai(prompt: str, model: str = "gpt-5-nano", max_tokens: int = 512) -> str:
    """
    Call OpenAI-compatible API (if `openai` is installed and OPENAI_API_KEY is present).
    This function is optional — if no API key / package available, the caller may use `use_mock=True`.
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package is required for real model calls") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()


def _mock_model_response_for_query(web_docs: List[Dict[str, Any]]) -> List[str]:
    """
    Simple deterministic mock: return 'N' for every doc if no other signal.
    This is used when `use_mock=True`.
    """
    return ["N" for _ in web_docs]


def merge_offline_with_web(
    offline_path: str = "data/offline/offline-summaries.json",
    web_cache_path: str = "cache/web_results.json",
    output_path: Optional[str] = None,
    model: str = "gpt-5-nano",
    use_mock: bool = True,
    num_queries: int = 185,
):
    """
    For i in range(num_queries):
      - load offline summary for query i
      - load web docs for query i from web cache
      - send both to model (or mock) which must return for each web doc one of:
            N        -> Not entailed (skip)
            EP:<idx> -> Entailed to pre-existing perspective index (0-based)
            EN:<text>-> Entailed as new perspective (text follows)
      - merge accordingly:
            N -> skip
            EP -> append doc url to that perspective's evidence list
            EN -> add a new perspective entry with the text and evidence url

    The function writes the merged offline summaries to `output_path` (or
    overwrites the input file if `output_path` is None).
    """
    # Load files
    offline_file = Path(offline_path)
    web_file = Path(web_cache_path)
    if not offline_file.exists():
        raise FileNotFoundError(f"Offline file not found: {offline_file}")
    if not web_file.exists():
        raise FileNotFoundError(f"Web cache file not found: {web_file}")

    with open(offline_file, "r", encoding="utf-8") as f:
        offline_data = json.load(f)

    with open(web_file, "r", encoding="utf-8") as f:
        web_data = json.load(f)

    # offline_data expected to be a list of query summaries
    # web_data expected to be a mapping from query id -> list of web docs

    # Helper to find offline entry for query i
    def _find_offline_entry(i: int):
        # try keys
        for entry in offline_data:
            # support id as int or str or field 'query_id'
            if entry.get("id") == i or entry.get("id") == str(i) or entry.get("query_id") == i:
                return entry
        # fallback: use i as index if available
        if 0 <= i < len(offline_data):
            return offline_data[i]
        return None

    for i in range(num_queries):
        entry = _find_offline_entry(i)
        if entry is None:
            # no offline entry for this index
            continue

        # web docs: try multiple key formats
        web_docs = web_data.get(str(i)) or web_data.get(i) or web_data.get(entry.get("id")) or []
        if not web_docs:
            continue

        # Normalize perspectives: ensure list of dicts with 'text' and 'evidence_urls'
        perspectives = entry.get("perspectives")
        if perspectives is None:
            # try older keys
            perspectives = entry.get("perspective_list") or []

        normalized = []
        for p in perspectives:
            if isinstance(p, dict):
                text = p.get("text") or p.get("perspective") or p.get("persp") or ""
                evidence = p.get("evidence_urls") or p.get("evidence") or p.get("sources") or []
                normalized.append({"text": text, "evidence_urls": list(evidence)})
            else:
                # assume string
                normalized.append({"text": str(p), "evidence_urls": []})

        # Build prompt listing web docs and existing perspectives; ask model to return labels for each web doc in order
        docs_prompt_lines = []
        for idx, doc in enumerate(web_docs):
            # each web doc should have url and snippet/content
            url = doc.get("url") or doc.get("link") or doc.get("source") or ""
            title = doc.get("title", "")
            snippet = (doc.get("snippet") or doc.get("content") or "")[:1000]
            docs_prompt_lines.append(f"[{idx}] URL: {url}\nTitle: {title}\nSnippet: {snippet}")

        prompt = (
            f"You are given a query and a list of existing perspectives (short text). "
            f"For each web document below, decide whether it is: "
            f"N => Not entailed by any existing perspective; "
            f"EP:<index> => Entailed to a pre-existing perspective (provide 0-based index of that perspective); "
            f"EN:<short perspective text> => Entailed as a new perspective (provide the short perspective text).\n\n"
            f"Query ID: {entry.get('id')}\n"
            f"Existing perspectives:\n"
        )
        for p_idx, p in enumerate(normalized):
            prompt += f"[{p_idx}] {p['text']}\n"

        prompt += "\nWeb documents:\n"
        prompt += "\n---\n".join(docs_prompt_lines)

        prompt += (
            "\n\nOutput: For each web document (in the same order), return a single line with the format: "
            "<idx>: <LABEL>  where <LABEL> is one of N, EP:<index>, EN:<short text>. "
            "Do not output any other text."
        )

        # Call model
        if use_mock:
            model_out_lines = _mock_model_response_for_query(web_docs)
            # mock returns list of 'N' strings; convert to lines prefixed
            model_lines = [f"{idx}: {lab}" for idx, lab in enumerate(model_out_lines)]
        else:
            resp_text = _call_model_openai(prompt, model=model)
            # split into lines and keep non-empty
            model_lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]

        # Parse model lines
        for line in model_lines:
            # expected format: 'idx: label'
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            try:
                doc_idx = int(left.strip())
            except Exception:
                continue
            label = right.strip()
            if doc_idx < 0 or doc_idx >= len(web_docs):
                continue

            webdoc = web_docs[doc_idx]
            url = webdoc.get("url") or webdoc.get("link") or webdoc.get("source") or None

            if label == "N":
                # skip
                continue
            elif label.startswith("EP"):
                # EP:<index>
                parts = label.split(":", 1)
                if len(parts) == 2:
                    try:
                        p_index = int(parts[1].strip())
                    except Exception:
                        continue
                    if 0 <= p_index < len(normalized):
                        if url:
                            normalized[p_index]["evidence_urls"].append(url)
            elif label.startswith("EN"):
                # EN:<text>
                parts = label.split(":", 1)
                new_text = parts[1].strip() if len(parts) == 2 else ""
                new_entry = {"text": new_text or webdoc.get("title") or webdoc.get("snippet") or "", "evidence_urls": []}
                if url:
                    new_entry["evidence_urls"].append(url)
                normalized.append(new_entry)

        entry["perspectives"] = normalized

    # Save merged offline file
    out_path = Path(output_path) if output_path else offline_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(offline_data, f, indent=2, ensure_ascii=False)

    return str(out_path)


if __name__ == "__main__":
    # Example run
    out = merge_offline_with_web(use_mock=True)
    print("Merged offline summaries written to:", out)
