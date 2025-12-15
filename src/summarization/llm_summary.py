import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field, field_validator
from typing import List
from outlines.models import from_transformers

# Module-level cache for model and generator
_model_cache = {}
_tokenizer_cache = {}
_outlines_model_cache = {}


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
            dtype=torch.float16,
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

def summarize_query(query: str, merged_corpus: list):
    """
    Generate multi-perspective summary using Llama-3.2-3B-Instruct with constrained JSON decoding.
    
    Args:
        query: the query/topic
        merged_corpus: list of documents with id, content, and score
    
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
    # Only require docs; the model generates claims itself
    if not merged_corpus:
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
    
    print("================================ CORPUS TEXT =================================")
    print(corpus_text)
    print("================================ CORPUS TEXT =================================")

    # Create prompt for multi-perspective summarization
    prompt = f"""Given the query and documents, create a multi-perspective summary with exactly 2 claims (one positive, one negative).

Query: {query}

Documents:
{corpus_text}

Rules:
1. IGNORE any documents that are clearly off-topic or irrelevant to the query - only cite documents that directly address the query's subject matter.
2. Include both a positive claim and a negative claim in response to the query
3. Each perspective MUST be a ONE-SENTENCE summary - DO NOT copy entire paragraphs from documents
4. Each perspective text should be concise (max 20-30 words) and summarize the key point
5. Each perspective MUST reference specific document IDs that support it - DO NOT use empty evidence_docs
6. Group related perspectives under the same claim
7. Ensure all document IDs used are from the provided documents
8. Each document ID can only be used ONCE across the entire summary. Different documents must support opposing viewpoints.
9. Prefer using multiple distinct documents for each claim; when available, aim for two or more distinct docs per claim, but prioritize validity and relevance.

Generate the JSON output now:"""

    # Retry loop for generation
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # Use constrained generation to enforce JSON schema
            result = model(prompt, MultiPerspectiveSummary, max_new_tokens=1500, temperature=0.1, top_p=0.8)
            
            print("================================ GENERATED RESPONSE =================================")
            print(result)
            print("================================ GENERATED RESPONSE =================================")
            
            # Parse JSON string result
            if isinstance(result, str):
                result_dict = json.loads(result)
                # Validate through Pydantic to enforce constraints
                summary_obj = MultiPerspectiveSummary(**result_dict)
                return [claim.model_dump() for claim in summary_obj.summaries]
            elif hasattr(result, 'summaries'):
                # If already a Pydantic object, return it
                return [claim.model_dump() for claim in result.summaries]
            else:
                return []

        except Exception as e:
            print(f"GENERATION ATTEMPT {attempt + 1}/{max_retries} FAILED: {e}")
            if attempt == max_retries - 1:
                # All attempts failed, return fallback
                print(f"All {max_retries} attempts failed. Returning fallback summary.")
                fallback_ids = [doc['id'] for doc in merged_corpus[:min(3, len(merged_corpus))]]
                return [
                    {
                        "claim": "Positive claim",
                        "perspectives": [
                            {
                                "text": f"Error generating summary: {str(e)[:100]}",
                                "evidence_docs": fallback_ids
                            }
                        ]
                    },
                    {
                        "claim": "Negative claim",
                        "perspectives": [
                            {
                                "text": f"Error generating summary: {str(e)[:100]}",
                                "evidence_docs": fallback_ids
                            }
                        ]
                    }
                ]

