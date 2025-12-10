import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Module-level cache for model and tokenizer
_model_cache = {}
_tokenizer_cache = {}

def _load_model(model_name: str, hf_token: str):
    """Load model and tokenizer with caching."""
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
            dtype=torch.float16
        ).to("cuda")
    
    return _model_cache[model_name], _tokenizer_cache[model_name]

def summarize_query(query: str, merged_corpus: list, claims: list):
    """
    Generate multi-perspective summary using Llama-3.1-8B-Instruct
    
    Args:
        query: the query/topic
        merged_corpus: list of documents with id, content, and score
        claims: list of 2 claims for different perspectives
    
    Returns:
        list: multi-perspective summary with structure:
        [
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
    """
    if not merged_corpus or len(claims) < 2:
        return []
    
    # Load model and tokenizer (cached on first call)
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    model, tokenizer = _load_model(model_name, HF_TOKEN)

    # Format corpus for the prompt
    corpus_text = "\n".join([
        f"[Doc {doc['id']}]: {doc.get('content', '')}"
        # f"[Doc {doc['id']}]: {doc.get('content', '')[:300]}"  # Limit content length
        for i, doc in enumerate(merged_corpus)
    ])
    
    print("================================ CORPUS TEXT =================================")
    print(corpus_text)
    print("================================ CORPUS TEXT =================================")

    # Create prompt for multi-perspective summarization
    prompt = f"""Given the query and documents, summarize the perspectives in the documents to the query together with their reference.

Query: {query}

Documents:
{corpus_text}

The summarization should adhere to the following rules:
1. The summarization should include both a positive claim and a negative claim in response to the query.
2. Each document corresponds to exactly one perspective.
3. Each perspective should be a coherent, one-sentence summary of the associated document.
4. The perspectives should not overlap with each other.
5. The number of perspectives for each claim may exceed one.
6. The number of supporting documents of each perspective may exceed one.
7. The summary content should be closely related to the query.
8. The output should be in the JSON format as below:
[
    {{
        "claim": "Positive claim in response to the query",
        "perspectives": [
            {{"text": "Perspective 1 supporting the positive claim", "evidence_docs": [doc_1, doc_2, ...]}},
            {{"text": "Perspective 2 supporting the positive claim", "evidence_docs": [doc_3, doc_4, ...]}},
            ...
        ]
    }},
    {{
        "claim": "Negative claim in response to the query",
        "perspectives": [
            {{"text": "Perspective 1 supporting the negative claim", "evidence_docs": [doc_5, doc_6, ...]}},
            {{"text": "Perspective 2 supporting the negative claim", "evidence_docs": [doc_7, doc_8, ...]}},
            ...
        ]
    }}
]

Respond ONLY with valid JSON, no additional text."""
    
    # print("prompt:", prompt)

    try:
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=850,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        # Decode only the newly generated tokens (exclude the prompt tokens)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]

        response_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ) if generated_ids.numel() else ""

        # print("================================ GENERATED RESPONSE =================================")
        # print(response_text)
        # print("================================ GENERATED RESPONSE =================================")

        def extract_first_json_array(text):
            start = text.find('[')
            if start == -1:
                return None

            depth = 0
            for i in range(start, len(text)):
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            return json.loads(candidate)
                        except:
                            return None
            return None

        # Try primary extraction
        clean_json = extract_first_json_array(response_text)
        if clean_json is not None:
            return clean_json

        # Try code-fenced blocks second
        fenced = re.findall(r'```(.*?)```', response_text, re.DOTALL)
        for block in fenced:
            try:
                return json.loads(block.strip())
            except:
                pass

        # Final fallback
        fallback_ids = [doc['id'] for doc in merged_corpus[:3]]
        return [
            {
                "claim": claims[0],
                "perspective": response_text,
                "evidence_docs": fallback_ids
            },
            {
                "claim": claims[1],
                "perspective": response_text,
                "evidence_docs": fallback_ids
            }
        ]

    except Exception as e:
        print("GENERATION FAILED:", e)
        return [
            {
                "claim": claims[0],
                "perspective": f"Error: {e}",
                "evidence_docs": []
            },
            {
                "claim": claims[1],
                "perspective": f"Error: {e}",
                "evidence_docs": []
            }
        ]

