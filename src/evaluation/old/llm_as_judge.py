"""
LLM-as-Judge Evaluation Module (PerSphere Implementation)

Evaluates multi-perspective summaries using the PerSphere "Gold Standard" methodology.
Gold references are read from data.jsonl by index (same order as offline-summaries.json).
Prompt uses Criteria 1-5 (Quality & Accuracy) per paper Section 4.6.
"""

import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_gold_reference(index: int, gold_file_path: str = "data/theperspective/data.jsonl") -> dict:
    """Retrieves the gold reference at the given index from the JSONL file."""
    try:
        with open(gold_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    item = json.loads(line.strip())
                    return {
                        "query": item.get("title", ""),
                        "perspectives": [
                            {
                                "claim": item.get("t1", ""),
                                "points": item.get("response1", []),
                                "evidence_ids": item.get("favor_ids", [])
                            },
                            {
                                "claim": item.get("t2", ""),
                                "points": item.get("response2", []),
                                "evidence_ids": item.get("against_ids", [])
                            }
                        ]
                    }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading gold data: {e}")
    return None

def llm_score_summary(summary: dict, index: int = None, reference: dict = None, model: str = "gpt-5-nano-2025-08-07") -> dict:
    """
    Evaluate summary quality using the PerSphere Prompt (Criteria 1-5).
    
    Args:
        summary: AI-generated summary dict with 'query' and 'perspectives'.
        index: Index in offline-summaries.json (for matching gold reference).
        reference: Optional gold reference; if None, retrieves by index.
        model: OpenAI model to use.
    
    Returns:
        dict with 'total_score' (1-10), 'explanation', and 'raw_response'.
    """
    if not reference and index is not None:
        reference = get_gold_reference(index)
    if not reference:
        return {"error": "Gold standard not found for this query.", "total_score": 0}

    gold_str = json.dumps(reference, indent=2)
    response_str = json.dumps(summary, indent=2)

    prompt = f"""Please serve as an impartial judge and assess the quality of a summarization performed by an AI assistant. You should strictly evaluate it with the golden summarization but not your own knowledge or assumption. We will provide both the ideal summarization ("golden summarization") and the AI's response.

Your evaluation should adhere to the following requirements:
1. The perspectives generated should accurately reflect the claims.
2. The references provided should support the corresponding perspectives.
3. Each perspective should be distinct and free from irrelevant information or overlap with others.
4. The perspectives should include content from the golden summarization (not necessarily all).
5. The content of the response should not be fabricated or expanded; it should not include anything that does not appear in the golden summarization.

Considering these factors, please begin your evaluation with a brief explanation, aiming for maximum objectivity. After providing your explanation, rate the response on a scale from 1 to 10, using this format: "Rating: [[number]]" (for example, "Rating: [[5]]").

Golden Summarization: {gold_str}
Response: {response_str}"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    client = OpenAI(api_key=api_key)
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as e:
        return {"error": str(e), "total_score": 0}

    match = re.search(r"Rating:\s*\[\[(\d+)\]\]", response_text, re.IGNORECASE)
    score = int(match.group(1)) if match else 0
    
    return {
        "total_score": score,
        "explanation": response_text.split("Rating:")[0].strip(),
        "raw_response": response_text
    }