import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def summarize_query(query: str, merged_corpus: list, claims: list):
    """
    Generate multi-perspective summary using Llama-3.1-8B-Instruct
    
    Args:
        query: the query/topic
        merged_corpus: list of documents with id, content, and score
        claims: list of 2 claims for different perspectives
    
    Returns:
        dict: multi-perspective summary with structure:
        {
            "query": str,
            "perspectives": [
                {
                    "claim": str,
                    "perspective": str,
                    "evidence_docs": list of doc ids
                },
                {
                    "claim": str,
                    "perspective": str,
                    "evidence_docs": list of doc ids
                }
            ]
        }
    """
    if not merged_corpus or len(claims) < 2:
        return {
            "query": query,
            "perspectives": []
        }
    
    # Load model, token, and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    HF_TOKEN = os.getenv("HF_TOKEN")
    # HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None
    ).to("cuda")
    
    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        temperature=0.3,
        top_p=0.9
    )
    
    # Format corpus for the prompt
    corpus_text = "\n".join([
        f"[Doc {i}]: {doc.get('content', '')[:300]}"  # Limit content length
        for i, doc in enumerate(merged_corpus)
    ])
    
    # Create prompt for multi-perspective summarization
    prompt = f"""Based on the following query and documents, generate a multi-perspective summary with exactly 2 perspectives.

Query: {query}

Claims to consider:
1. {claims[0]}
2. {claims[1]}

Documents:
{corpus_text}

Generate a response in JSON format with exactly this structure:
{{
    "perspectives": [
        {{
            "claim": "First claim",
            "perspective": "A perspective supporting or relating to the first claim",
            "evidence_docs": [list of document indices used by perspective used to support the claim]
        }},
        {{
            "claim": "Second claim",
            "perspective": "A perspective supporting or relating to the second claim",
            "evidence_docs": [list of document indices used by perspective used to support the claim]
        }}
    ]
}}

Only respond with valid JSON, no additional text."""

    try:
        # Generate response
        response = pipe(prompt)
        response_text = response[0]['generated_text']
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            summary_data = json.loads(json_match.group())
        else:
            # Fallback if JSON extraction fails
            summary_data = {
                "perspectives": [
                    {
                        "claim": claims[0],
                        "perspective": response_text,
                        "evidence_docs": list(range(min(len(merged_corpus), 3)))
                    },
                    {
                        "claim": claims[1],
                        "perspective": response_text,
                        "evidence_docs": list(range(min(3, len(merged_corpus))))
                    }
                ]
            }
        
        summary_data["query"] = query
        return summary_data
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Return fallback structure
        return {
            "query": query,
            "perspectives": [
                {
                    "claim": claims[0],
                    "perspective": f"Could not generate perspective due to error: {str(e)}",
                    "evidence_docs": []
                },
                {
                    "claim": claims[1],
                    "perspective": f"Could not generate perspective due to error: {str(e)}",
                    "evidence_docs": []
                }
            ]
        }