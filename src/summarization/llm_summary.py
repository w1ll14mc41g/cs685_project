import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    
    # Load model, token, and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is REQUIRED but not available!")

    # device = 0 if torch.cuda.is_available() else -1
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"

    HF_TOKEN = os.getenv("HF_TOKEN")
    # HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual token
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        dtype=torch.float16
    ).to("cuda")

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

Claims:
1. {claims[0]}
2. {claims[1]}

Documents:
{corpus_text}

The summarization should follow the requirements:
1. Each summarization should include both positive and negative claims, only 2 claims total.
2. The are different perspectives to the claims with their reference. The perspective should not overlap to each other.
3. The summary content should be closely related to the query.
4. The reference of each perspective may exceed one.
5. The output should be in the JSON format as below:
[
    {{
        "claim": "First claim",
        "perspective": "Perspectives supporting or relating to the first claim",
        "evidence_docs": [list of document indices used by perspective used to support the claim]
    }},
    {{
        "claim": "Second claim",
        "perspective": "Perspectives supporting or relating to the second claim",
        "evidence_docs": [list of document indices used by perspective used to support the claim]
    }}
]

Only respond with valid JSON, no additional text."""
    
    # print("prompt:", prompt)

    try:
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
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