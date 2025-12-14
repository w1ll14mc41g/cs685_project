import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from outlines.models import from_transformers

# Module-level cache for model and generator
_model_cache = {}
_tokenizer_cache = {}
_outlines_model_cache = {}


def _sanitize_result_dict(result_dict: Dict[str, Any], available_doc_ids: List[int]) -> Dict[str, Any]:
    """Post-process generated JSON to enforce uniqueness when possible, but allow repetition to preserve both claims.

    Strategy:
    1. Extract document IDs from perspective text if evidence_docs is empty (model often writes "Doc X" in text)
    2. First pass: Try to enforce global doc uniqueness (a doc ID appears only once)
    3. If a claim ends up empty after filtering for uniqueness, second pass: Allow that claim to reuse docs
    4. This preserves document uniqueness when possible, but prioritizes having 2 complete claims

    Returns sanitized summaries with at least 2 claims if possible.
    """
    import re
    
    summaries = result_dict.get("summaries", []) or []
    available_set = {d for d in available_doc_ids if d is not None}
    
    # Helper function to extract doc IDs from text
    def extract_doc_ids_from_text(text: str) -> List[int]:
        """Extract document IDs mentioned in text like 'Doc 123' or '(Doc 456)'."""
        if not text:
            return []
        matches = re.findall(r'[Dd]oc\s*(\d+)', text)
        doc_ids = [int(m) for m in matches]
        return [d for d in doc_ids if d in available_set]
    
    # First pass: try to enforce global uniqueness
    seen_docs = set()
    first_pass_summaries = []
    claims_with_empty_perspectives = []

    for claim_idx, claim in enumerate(summaries):
        claim_text = claim.get("claim") if isinstance(claim, dict) else getattr(claim, "claim", None)
        perspectives = claim.get("perspectives", []) if isinstance(claim, dict) else getattr(claim, "perspectives", [])
        new_perspectives = []

        for p in perspectives:
            p_text = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
            p_docs = p.get("evidence_docs", []) if isinstance(p, dict) else getattr(p, "evidence_docs", [])
            
            # If evidence_docs is empty, try to extract from text
            if not p_docs:
                p_docs = extract_doc_ids_from_text(p_text)
            
            # Keep only valid, unseen doc IDs
            unique_docs = [d for d in p_docs if d in available_set and d not in seen_docs]

            # If no valid docs found, assign a fallback available doc that hasn't been seen
            if not unique_docs and available_set:
                for fallback_id in available_set:
                    if fallback_id not in seen_docs:
                        unique_docs = [fallback_id]
                        break

            if not unique_docs:
                continue

            seen_docs.update(unique_docs)
            new_perspectives.append({"text": p_text, "evidence_docs": unique_docs})

        if new_perspectives:
            first_pass_summaries.append({"claim": claim_text, "perspectives": new_perspectives})
        else:
            # This claim lost all perspectives in first pass; we'll try to recover it
            claims_with_empty_perspectives.append(claim_idx)

    # If we have 2+ claims, we're good; return first pass result
    if len(first_pass_summaries) >= 2:
        return {"summaries": first_pass_summaries}

    # Second pass: if we lost claims due to uniqueness, allow document reuse for missing claims
    if claims_with_empty_perspectives:
        # Keep successful claims from first pass and track their docs
        seen_docs = set()
        for claim in first_pass_summaries:
            for p in claim.get("perspectives", []):
                seen_docs.update(p.get("evidence_docs", []))
        
        # Rebuild only the failed claims with document reuse allowed
        second_pass_summaries = list(first_pass_summaries)  # Start with successful claims

        for claim_idx in claims_with_empty_perspectives:
            if claim_idx >= len(summaries):
                continue
            
            claim = summaries[claim_idx]
            claim_text = claim.get("claim") if isinstance(claim, dict) else getattr(claim, "claim", None)
            perspectives = claim.get("perspectives", []) if isinstance(claim, dict) else getattr(claim, "perspectives", [])
            new_perspectives = []

            for p in perspectives:
                p_text = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
                p_docs = p.get("evidence_docs", []) if isinstance(p, dict) else getattr(p, "evidence_docs", [])
                
                # If evidence_docs is empty, try to extract from text
                if not p_docs:
                    p_docs = extract_doc_ids_from_text(p_text)
                
                # Allow reuse for failed claims - use any valid doc
                unique_docs = [d for d in p_docs if d in available_set]

                # If no valid docs found, assign a fallback available doc (allows reuse)
                if not unique_docs and available_set:
                    # Try to find an unseen one first, but if none exist, reuse any
                    unused_docs = [d for d in available_set if d not in seen_docs]
                    if unused_docs:
                        unique_docs = [unused_docs[0]]
                    else:
                        # All docs are used, pick any available doc (allows reuse)
                        unique_docs = [list(available_set)[0]]

                if unique_docs:
                    new_perspectives.append({"text": p_text, "evidence_docs": unique_docs})

            if new_perspectives:
                second_pass_summaries.append({"claim": claim_text, "perspectives": new_perspectives})

        if len(second_pass_summaries) >= 2:
            return {"summaries": second_pass_summaries}
    
    # Fallback: return whatever we got from first pass, even if < 2 claims
    return {"summaries": first_pass_summaries} if first_pass_summaries else {}

# Define the expected JSON schema using Pydantic
class Perspective(BaseModel):
    text: str = Field(description="One-sentence perspective summary")
    evidence_docs: List[int] = Field(description="List of document IDs supporting this perspective")

    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, text):
        """Ensure perspective text is not empty or just whitespace."""
        if not text or not text.strip():
            raise ValueError("Perspective text cannot be empty")
        return text

    @field_validator('evidence_docs')
    @classmethod
    def validate_evidence_docs_not_empty(cls, evidence_docs):
        """Ensure each perspective has at least one evidence document."""
        if not evidence_docs or len(evidence_docs) == 0:
            raise ValueError("Each perspective must have at least one evidence document")
        return evidence_docs

class Claim(BaseModel):
    claim: str = Field(description="The claim in response to the query")
    perspectives: List[Perspective] = Field(description="List of perspectives supporting this claim")

    @field_validator('claim')
    @classmethod
    def validate_claim_not_empty(cls, claim):
        """Ensure claim text is not empty or just whitespace."""
        if not claim or not claim.strip():
            raise ValueError("Claim text cannot be empty")
        return claim

class MultiPerspectiveSummary(BaseModel):
    summaries: List[Claim] = Field(description="List of claims with their perspectives", min_items=2, max_items=2)

def _load_model(model_name: str, hf_token: str):
    """Load transformers model and tokenizer with caching."""
    if model_name not in _model_cache:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is REQUIRED but not available!")
        
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        _model_cache[model_name] = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    return _model_cache[model_name], _tokenizer_cache[model_name]

def _get_outlines_model(model_name: str, hf_token: str):
    """Get outlines model wrapper with caching."""
    if model_name not in _outlines_model_cache:
        hf_model, tokenizer = _load_model(model_name, hf_token)
        # Wrap the transformers model with outlines
        _outlines_model_cache[model_name] = from_transformers(hf_model, tokenizer)
    
    return _outlines_model_cache[model_name]

def summarize_query(query: str, merged_corpus: list, claims: list):
    """
    Generate multi-perspective summary using Llama-3.2-3B-Instruct with constrained JSON decoding.
    
    Args:
        query: the query/topic
        merged_corpus: list of documents with id, content, and score
        claims: list of 2 claims for different perspectives
    
    Returns:
        list: multi-perspective summary with structure:
        [
            {
                "claim": str,
                "perspectives": [
                    {"text": str, "evidence_docs": list of doc ids}
                ]
            }
        ]
    """
    if not merged_corpus or len(claims) < 2:
        return []
    
    # Load outlines model (cached on first call)
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    try:
        model = _get_outlines_model(model_name, HF_TOKEN)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    # Format corpus for the prompt
    corpus_text = "\n".join([
        f"[Doc {doc['id']}]: {doc.get('content', '')}"
        for doc in merged_corpus
    ])
    available_doc_ids = [doc.get('id') for doc in merged_corpus]
    
    print("================================ CORPUS TEXT =================================")
    print(corpus_text)
    print("================================ CORPUS TEXT =================================")

    # Create prompt for multi-perspective summarization
    prompt = f"""Given the query and documents, create a multi-perspective summary with exactly 2 claims (one positive, one negative).

Query: {query}

Documents:
{corpus_text}

Rules:
1. Include both a positive claim and a negative claim in response to the query
2. Each perspective MUST be a ONE-SENTENCE summary - DO NOT copy entire paragraphs from documents
3. Each perspective text should be concise (max 20-30 words) and summarize the key point
4. Each perspective MUST reference specific document IDs that support it - DO NOT use empty evidence_docs
5. Group related perspectives under the same claim
6. Ensure all document IDs used are from the provided documents
7. CRITICAL: Each document ID can only be used ONCE across the entire summary. Different documents must support opposing viewpoints.
8. Prefer using multiple distinct documents for each claim; when available, aim for two or more distinct docs per claim, but prioritize validity and relevance.
9. IGNORE any documents that are clearly off-topic or irrelevant to the query - only cite documents that directly address the query's subject matter.


Generate the JSON output now:"""

    last_json = None  # best-effort parsed JSON from latest attempt
    best_json_with_2_claims = None  # track best result with at least 2 claims

    # Retry loop for generation
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # Use constrained generation to enforce JSON schema
            result = model(prompt, MultiPerspectiveSummary, max_new_tokens=1500, temperature=0.1, top_p=0.9)
            
            print("================================ GENERATED RESPONSE =================================")
            print(result)
            print("================================ GENERATED RESPONSE =================================")
            
            # Parse JSON string result
            if isinstance(result, str):
                result_dict = json.loads(result)
                sanitized = _sanitize_result_dict(result_dict, available_doc_ids)
                if sanitized and len(sanitized.get("summaries", [])) >= 2:
                    last_json = sanitized
                    # Track successfully sanitized results with 2+ claims
                    best_json_with_2_claims = sanitized
                    # Validate through Pydantic to enforce constraints
                    summary_obj = MultiPerspectiveSummary(**sanitized)
                    return [claim.model_dump() for claim in summary_obj.summaries]
                elif sanitized:
                    last_json = sanitized
                    summary_obj = MultiPerspectiveSummary(**sanitized)
                    return [claim.model_dump() for claim in summary_obj.summaries]
                else:
                    # Sanitization removed all perspectives - don't save as fallback
                    last_json = None
                    raise ValueError("Sanitization removed all perspectives")
            elif hasattr(result, 'summaries'):
                # If already a Pydantic object, sanitize then return
                summaries = [claim.model_dump() for claim in result.summaries]
                sanitized = _sanitize_result_dict({"summaries": summaries}, available_doc_ids)
                if sanitized and len(sanitized.get("summaries", [])) >= 2:
                    last_json = sanitized
                    # Track successfully sanitized results with 2+ claims
                    best_json_with_2_claims = sanitized
                    summary_obj = MultiPerspectiveSummary(**sanitized)
                    return [claim.model_dump() for claim in summary_obj.summaries]
                elif sanitized:
                    last_json = sanitized
                    summary_obj = MultiPerspectiveSummary(**sanitized)
                    return [claim.model_dump() for claim in summary_obj.summaries]
                else:
                    # Sanitization removed all perspectives - don't save as fallback
                    last_json = None
                    raise ValueError("Sanitization removed all perspectives")
            else:
                return []

        except Exception as e:
            print(f"GENERATION ATTEMPT {attempt + 1}/{max_retries} FAILED: {e}")
            if attempt == max_retries - 1:
                # Last attempt failed; prioritize returning result with 2+ claims
                result_to_use = best_json_with_2_claims if best_json_with_2_claims else last_json
                if result_to_use and "summaries" in result_to_use:
                    summaries = result_to_use.get("summaries", [])
                    normalized = []
                    for claim in summaries:
                        if hasattr(claim, "model_dump"):
                            normalized.append(claim.model_dump())
                        else:
                            normalized.append(claim)
                    if len(normalized) >= 2:
                        return normalized

                # No parsed JSON available, return fallback
                print(f"All {max_retries} attempts failed. Returning fallback summary.")
                fallback_ids = [doc['id'] for doc in merged_corpus[:min(3, len(merged_corpus))]]
                return [
                    {
                        "claim": claims[0] if len(claims) > 0 else "Positive claim",
                        "perspectives": [
                            {
                                "text": f"Error generating summary: {str(e)[:100]}",
                                "evidence_docs": fallback_ids
                            }
                        ]
                    },
                    {
                        "claim": claims[1] if len(claims) > 1 else "Negative claim",
                        "perspectives": [
                            {
                                "text": f"Error generating summary: {str(e)[:100]}",
                                "evidence_docs": fallback_ids
                            }
                        ]
                    }
                ]

