"""
LLM-as-Judge Evaluation Module (Improved Implementation)

Evaluates multi-perspective summaries using a fair 5-criteria rubric (10 points total)
that works equally for offline-only and web-augmented summaries.
Gold references are matched by query text from data.jsonl.
"""

import os
import json
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


# Pydantic models for structured output
class Scores(BaseModel):
    """Individual criterion scores (total is computed automatically)."""
    criterion_1_claim_relevance: int = Field(ge=0, le=2, description="Claim relevance score (0-2)")
    criterion_2_perspective_claim_alignment: int = Field(ge=0, le=2, description="Perspective-claim alignment score (0-2)")
    criterion_3_perspective_distinctness: int = Field(ge=0, le=2, description="Perspective distinctness score (0-2)")
    criterion_4_coverage_of_core_arguments: int = Field(ge=0, le=2, description="Coverage of core arguments score (0-2)")
    criterion_5_factual_grounding: int = Field(ge=0, le=2, description="Factual grounding score (0-2)")


class Explanations(BaseModel):
    """Explanations for each criterion."""
    criterion_1: str = Field(description="Explanation for claim relevance")
    criterion_2: str = Field(description="Explanation for perspective-claim alignment")
    criterion_3: str = Field(description="Explanation for perspective distinctness")
    criterion_4: str = Field(description="Explanation for coverage of core arguments")
    criterion_5: str = Field(description="Explanation for factual grounding")


class EvaluationResponse(BaseModel):
    """Complete evaluation response with scores and explanations."""
    scores: Scores
    explanations: Explanations


def get_gold_reference_by_query(query: str, gold_file_path: str = "data/theperspective/data.jsonl") -> Optional[List[Dict]]:
    """
    Retrieves the gold reference by matching query text (title field) from the JSONL file.
    
    Returns the gold reference in the same format as the actual summaries:
    [
        {
            "claim": "...",
            "perspectives": [
                {
                    "text": "...",
                    "evidence_docs": [...]
                }
            ]
        }
    ]
    """
    try:
        with open(gold_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("title", "").strip() == query.strip():
                    # Convert to match the actual summary format
                    gold_summary = []
                    
                    # First claim (t1) with perspectives from response1
                    claim1_perspectives = []
                    for point_text in item.get("response1", []):
                        claim1_perspectives.append({
                            "text": point_text,
                            "evidence_docs": item.get("favor_ids", [])
                        })
                    gold_summary.append({
                        "claim": item.get("t1", ""),
                        "perspectives": claim1_perspectives
                    })
                    
                    # Second claim (t2) with perspectives from response2
                    claim2_perspectives = []
                    for point_text in item.get("response2", []):
                        claim2_perspectives.append({
                            "text": point_text,
                            "evidence_docs": item.get("against_ids", [])
                        })
                    gold_summary.append({
                        "claim": item.get("t2", ""),
                        "perspectives": claim2_perspectives
                    })
                    
                    return gold_summary
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading gold data: {e}")
    return None


def extract_web_docs_from_merged(query: str, merged_file_path: str) -> List[Dict]:
    """
    Extract web docs (URL entries) for a query from merged corpus file.
    
    Args:
        query: Query text to match
        merged_file_path: Path to merged corpus JSON file
        
    Returns:
        List of web docs where id is a string starting with "https://"
    """
    try:
        with open(merged_file_path, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
            
        for entry in merged_data:
            if entry.get("query", "").strip() == query.strip():
                merged_docs = entry.get("merged", [])
                web_docs = []
                for doc in merged_docs:
                    doc_id = doc.get("id", "")
                    # Web docs have string IDs starting with "https://"
                    if isinstance(doc_id, str) and doc_id.startswith("https://"):
                        web_docs.append({
                            "id": doc_id,
                            "content": doc.get("content", "")
                        })
                return web_docs
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading merged corpus: {e}")
    return []


def is_error_summary(summary: Union[List, Dict], is_merged: bool = False) -> bool:
    """
    Detect error summaries based on error patterns.
    
    Args:
        summary: Summary to check (can be list of claims or dict with summary field)
        is_merged: True if from merged summaries, False if from offline summaries
        
    Returns:
        True if summary contains error patterns, False otherwise
    """
    # Handle different summary formats
    if isinstance(summary, dict):
        summary_list = summary.get("summary", summary.get("summaries", []))
    else:
        summary_list = summary
    
    if not isinstance(summary_list, list):
        return False
    
    # Check each claim's perspectives for error patterns
    for claim in summary_list:
        if not isinstance(claim, dict):
            continue
        perspectives = claim.get("perspectives", [])
        for perspective in perspectives:
            if not isinstance(perspective, dict):
                continue
            text = perspective.get("text", "")
            
            if is_merged:
                # Merged summary error patterns
                if "JSON parse errors 5+ times (prompt too long)" in text:
                    return True
                if "All 10 generation attempts failed" in text:
                    return True
            else:
                # Offline summary error patterns
                if text.startswith("Error generating summary:"):
                    return True
    
    return False


def llm_score_summary(
    summary: Union[List, Dict],
    query: str,
    reference: Optional[List[Dict]] = None,
    web_docs: Optional[List[Dict]] = None,
    model: str = "gpt-5-nano-2025-08-07"
) -> Dict:
    """
    Evaluate summary quality using the improved 5-criteria rubric.
    
    Uses the Responses API with Structured Outputs via Pydantic models for guaranteed
    schema adherence. Structured Outputs ensures valid JSON that matches the schema.
    
    Args:
        summary: AI-generated summary (list of claims or dict with summary field)
        query: The original query text
        reference: Optional gold reference (list format matching summary structure); if None, retrieves by query text
        web_docs: Optional list of web docs (for merged summaries)
        model: OpenAI model to use (must support Structured Outputs, e.g., gpt-4o-2024-08-06)
    
    Returns:
        dict with detailed scores per criterion, total_score, explanations, and raw_response.
        If the model refuses or an API error occurs, includes "error" field.
    """
    # Get gold reference if not provided
    if not reference:
        reference = get_gold_reference_by_query(query)
    if not reference:
        return {
            "error": "Gold standard not found for this query.",
            "total_score": 0,
            "criterion_1_claim_relevance": 0,
            "criterion_2_perspective_claim_alignment": 0,
            "criterion_3_perspective_distinctness": 0,
            "criterion_4_coverage_of_core_arguments": 0,
            "criterion_5_factual_grounding": 0
        }
    
    # Handle different summary formats
    if isinstance(summary, dict):
        summary_list = summary.get("summary", summary.get("summaries", []))
    else:
        summary_list = summary
    
    gold_str = json.dumps(reference, indent=2)
    response_str = json.dumps(summary_list, indent=2)
    
    # Build evidence context
    evidence_context = f"""Query: {query}

Gold Summary (for reference): {gold_str}"""
    
    if web_docs and len(web_docs) > 0:
        web_context = "\n".join([
            f"[Web Doc {i+1}]: {doc.get('content', '')[:200]}..." 
            for i, doc in enumerate(web_docs)
        ])
        evidence_context += f"\n\nAdditional Web Evidence:\n{web_context}"
    
    # Unified prompt (same for offline and online)
    prompt = f"""You are evaluating a multi-perspective summary. Judge the summary based on the criteria below, assigning points for each category.

{evidence_context}

Summary to Evaluate:
{response_str}

---

EVALUATION CRITERIA (10 points total):

1. CLAIM RELEVANCE (0-2 points)
   - Do the claims directly address the query?
   - Are they appropriately oppositional (pro/con)?
   - Scoring:
     - 2: Both claims clearly relevant and oppositional
     - 1: Claims somewhat relevant but unclear opposition
     - 0: Claims off-topic or not oppositional

2. PERSPECTIVE-CLAIM ALIGNMENT (0-2 points)
   - Does each perspective clearly support one of the stated claims?
   - Is the stance (pro/con) consistent?
   - Scoring:
     - 2: All perspectives clearly aligned with correct claim
     - 1: Most perspectives aligned, some ambiguous
     - 0: Perspectives misaligned or unclear which claim they support

3. PERSPECTIVE DISTINCTNESS (0-2 points)
   - Are perspectives non-overlapping and unique?
   - Does each present a different supporting argument?
   - Scoring:
     - 2: All perspectives distinct with no overlap
     - 1: Mostly distinct but some redundancy
     - 0: Heavy overlap or repetitive perspectives

4. COVERAGE OF CORE ARGUMENTS (0-2 points)
   - Does the summary capture key arguments from available evidence?
   - This includes arguments from the gold summary AND/OR valid new arguments from additional evidence
   - Scoring:
     - 2: Comprehensive coverage of core arguments (gold or novel)
     - 1: Partial coverage, missing some key arguments
     - 0: Poor coverage, misses most arguments

5. FACTUAL GROUNDING (0-2 points)
   - Are perspectives supported by the provided evidence?
   - Are there hallucinations or fabricated claims not present in any evidence?
   - Scoring:
     - 2: All perspectives clearly grounded in evidence, no fabrication
     - 1: Mostly grounded but some unsupported claims
     - 0: Significant hallucination or fabricated content

---

INSTRUCTIONS:
- Provide a brief explanation for each criterion
- Assign points for each criterion (0, 1, or 2)
- Return your evaluation in the specified structured format"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    client = OpenAI(api_key=api_key)
    
    try:
        # Use Responses API with Structured Outputs via Pydantic model
        # This guarantees schema adherence - no need for retries or fallback parsing
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": "You are evaluating a multi-perspective summary. Judge the summary based on the criteria provided, assigning points for each category."},
                {"role": "user", "content": prompt}
            ],
            text_format=EvaluationResponse,
	    reasoning={"effort": "low"}
        )
        
        # Check for refusal (Structured Outputs feature)
        if response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'message':
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'type') and content_item.type == 'refusal':
                                refusal_reason = getattr(content_item, 'refusal', 'Unknown reason')
                                raw_response = getattr(response, 'output_text', '') or json.dumps({"refusal": refusal_reason}, indent=2)
                                return {
                                    "error": f"Model refused to respond: {refusal_reason}",
                                    "total_score": 0,
                                    "criterion_1_claim_relevance": 0,
                                    "criterion_2_perspective_claim_alignment": 0,
                                    "criterion_3_perspective_distinctness": 0,
                                    "criterion_4_coverage_of_core_arguments": 0,
                                    "criterion_5_factual_grounding": 0,
                                    "raw_response": raw_response
                                }
        
        # Extract parsed output (guaranteed to match schema with Structured Outputs)
        parsed = response.output_parsed
        scores = parsed.scores
        explanations = parsed.explanations
        
        # Calculate total score by summing individual criterion scores
        total_score = (
            scores.criterion_1_claim_relevance +
            scores.criterion_2_perspective_claim_alignment +
            scores.criterion_3_perspective_distinctness +
            scores.criterion_4_coverage_of_core_arguments +
            scores.criterion_5_factual_grounding
        )
        
        # Build explanation text
        explanations_text = "\n\n".join([
            f"Criterion {i+1}: {getattr(explanations, f'criterion_{i+1}', '')}"
            for i in range(5)
        ])
        
        # Get raw response text for logging
        raw_response = getattr(response, 'output_text', '') or json.dumps(parsed.model_dump(), indent=2)
        
        return {
            "criterion_1_claim_relevance": scores.criterion_1_claim_relevance,
            "criterion_2_perspective_claim_alignment": scores.criterion_2_perspective_claim_alignment,
            "criterion_3_perspective_distinctness": scores.criterion_3_perspective_distinctness,
            "criterion_4_coverage_of_core_arguments": scores.criterion_4_coverage_of_core_arguments,
            "criterion_5_factual_grounding": scores.criterion_5_factual_grounding,
            "total_score": total_score,
            "explanations": {
                "criterion_1": explanations.criterion_1,
                "criterion_2": explanations.criterion_2,
                "criterion_3": explanations.criterion_3,
                "criterion_4": explanations.criterion_4,
                "criterion_5": explanations.criterion_5,
            },
            "raw_response": raw_response,
            "explanation": explanations_text
        }
        
    except Exception as e:
        # Handle API errors (network issues, rate limits, etc.)
        error_msg = str(e)
        return {
            "error": f"API error: {error_msg}",
            "total_score": 0,
            "criterion_1_claim_relevance": 0,
            "criterion_2_perspective_claim_alignment": 0,
            "criterion_3_perspective_distinctness": 0,
            "criterion_4_coverage_of_core_arguments": 0,
            "criterion_5_factual_grounding": 0,
            "raw_response": ""
        }
