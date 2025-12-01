"""
LLM-as-Judge Evaluation Module

Evaluates multi-perspective summaries using GPT with 5 criteria (0-2 each):
1. Claim-Perspective Alignment: Perspectives accurately reflect the claims
2. Evidence Support: References support the corresponding perspectives  
3. Perspective Distinctiveness: Each perspective is distinct, no overlap
4. Query Coverage: Perspectives comprehensively cover the query topic
5. Content Groundedness: Content is not fabricated or hallucinated

Since gold summaries are not yet available, we use intrinsic quality evaluation
where the LLM assesses internal consistency and reasonableness rather than
comparison to a reference. This maps the original criteria as follows:
- Criteria 1-3: Evaluated directly (claim alignment, evidence, distinctiveness)
- Criterion 4: "Cover golden summary" → "Cover query topic comprehensively"
- Criterion 5: "Not fabricated vs gold" → "Content appears factual/grounded"

Total score: 0-10 (sum of 5 criteria, each 0-2)
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()


def llm_score_summary(summary: dict, reference: dict = None, model: str = "gpt-4o-mini") -> dict:
    """
    Evaluate summary quality using LLM-as-Judge.
    
    Args:
        summary: Summary dict with 'query' and 'perspectives' 
        reference: Optional gold reference (for future use)
        model: OpenAI model to use
    
    Returns:
        dict with scores (0-2 each) and total_score (0-10)
    """
    query = summary.get("query", "")
    perspectives = summary.get("perspectives", [])
    
    if not query or not perspectives:
        return {"error": "Invalid summary", "total_score": 0}
    
    # Check for malformed perspectives (prompt leakage)
    for p in perspectives:
        if "Generate a response in JSON format" in p.get("perspective", ""):
            return {"error": "Malformed perspective", "total_score": 0}
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Format perspectives
    perspectives_text = ""
    for i, p in enumerate(perspectives, 1):
        perspectives_text += f"""
Perspective {i}:
- Claim: {p.get('claim', 'N/A')}
- Perspective: {p.get('perspective', 'N/A')}
- Evidence Document IDs: {p.get('evidence_docs', [])}
"""

    prompt = f"""Please serve as an impartial judge and assess the quality of a multi-perspective summarization performed by an AI assistant.

Since we do not have a gold reference summary, evaluate the summary based on its intrinsic quality. Your evaluation should adhere to the following 5 criteria, each rated on a scale from 0 to 2:
- 0: Completely unsatisfactory
- 1: Neutral/Partially satisfactory
- 2: Fully satisfactory

EVALUATION CRITERIA:

1. **Claim-Perspective Alignment**: The perspectives generated should accurately reflect the claims. Does each perspective logically support and elaborate on its associated claim?

2. **Evidence Support**: The references provided should support the corresponding perspectives. Are evidence document IDs provided, and does having references seem appropriate for each perspective?

3. **Perspective Distinctiveness**: Each perspective should be distinct and free from irrelevant information or overlap with others. Are the perspectives genuinely different viewpoints without redundancy?

4. **Query Coverage**: The perspectives should comprehensively cover the topic. Do the perspectives address the key aspects of the query/debate from multiple angles?

5. **Content Groundedness**: The content of the response should not be fabricated or expanded beyond reason. Does the content appear factual and well-grounded rather than containing hallucinations or nonsensical claims?

The sequence of perspectives does not need to match any particular order.
Format differences are not critical criteria.

---

QUERY: {query}

RESPONSE TO EVALUATE:
{perspectives_text}

---

Considering these factors, please begin your evaluation with a brief explanation for each criterion, aiming for maximum objectivity. After providing your explanation, rate each criterion and provide a total score.

Format your response as:
1. **Claim-Perspective Alignment**: [Explanation] Score: [0/1/2]
2. **Evidence Support**: [Explanation] Score: [0/1/2]
3. **Perspective Distinctiveness**: [Explanation] Score: [0/1/2]
4. **Query Coverage**: [Explanation] Score: [0/1/2]
5. **Content Groundedness**: [Explanation] Score: [0/1/2]

End with the total score using this format: "Rating: [[number]]" (for example, "Rating: [[7]]")
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800
    )
    
    response_text = response.choices[0].message.content
    
    # Parse scores
    result = {"raw_response": response_text}
    
    patterns = [
        ("claim_perspective_alignment", r"Claim-Perspective Alignment.*?Score:\s*(\d)"),
        ("evidence_support", r"Evidence Support.*?Score:\s*(\d)"),
        ("perspective_distinctiveness", r"Perspective Distinctiveness.*?Score:\s*(\d)"),
        ("query_coverage", r"Query Coverage.*?Score:\s*(\d)"),
        ("content_groundedness", r"Content Groundedness.*?Score:\s*(\d)"),
    ]
    
    for key, pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        result[key] = int(match.group(1)) if match else None
    
    # Extract total
    total_match = re.search(r"Rating:\s*\[\[(\d+)\]\]", response_text)
    if total_match:
        result["total_score"] = int(total_match.group(1))
    else:
        scores = [result.get(k) for k, _ in patterns]
        result["total_score"] = sum(s for s in scores if s is not None)
    
    return result