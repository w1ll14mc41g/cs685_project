import json
import os
import numpy as np
import statistics
from collections import defaultdict

"""
Calculate web metrics:
    New Relevant Doc Count - Number of web docs passing relevance filter per query
    Relevance Rate - relevant_count / total_web_retrieved - Retrieval precision

    Token Increase Rate - web_tokens / baseline_tokens

    For each metric, use bootstrapping to calculate a 95% confidence interval,
    also include important metrics such as mean, median, q1, q2, q3, q4 values.

    it should read in from data/valid-web/valid-web-{k}.json for k = 5, 10, and 20

    New Relevant Doc Count should be the number of relevance: "R"
    Relevance rate is the relevant count above divided by the corresponding k value

    for token increase rate it should read from data/merged-corpus/merged-{k}.json 's "content" of
    each query and compare it to the content within the same query that have "score" key representing
    the baseline contents, count each word as a token (split by space is sufficient)


Returns:
    all metric data saved in a file
"""
def bootstrap_ci(data, n_boot=1000, ci=95):
    """Compute bootstrap confidence interval."""
    if len(data) == 0:
        return None, None
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper

def count_tokens(text):
    """Count tokens by splitting on spaces."""
    return len(text.split())

def compute_metrics(k):
    # Load valid-web data
    valid_web_path = f"data/valid-web/valid-web-{k}.json"
    with open(valid_web_path, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    
    relevant_counts = []
    relevance_rates = []
    
    for query_data in valid_data:
        results = query_data['web_docs']['results']
        relevant_count = sum(1 for doc in results if doc.get('relevance') == 'R')
        relevant_counts.append(relevant_count)
        relevance_rates.append(relevant_count / k)
    
    # Load merged data
    merged_path = f"data/merged-corpus/merged-{k}.json"
    with open(merged_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    token_increase_rates = []
    
    for query_data in merged_data:
        merged_docs = query_data['merged']
        baseline_tokens = 0
        total_tokens = 0
        for doc in merged_docs:
            content = doc['content']
            total_tokens += count_tokens(content)
            if 'score' in doc:
                baseline_tokens += count_tokens(content)
        if baseline_tokens > 0:
            token_increase_rates.append(total_tokens / baseline_tokens)
    
    # Compute stats for each metric
    metrics = {}
    
    for name, values in [('New Relevant Doc Count', relevant_counts),
                         ('Relevance Rate', relevance_rates),
                         ('Token Increase Rate', token_increase_rates)]:
        if values:
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            q1, q2, q3, q4 = np.percentile(values, [25, 50, 75, 100])
            ci_lower, ci_upper = bootstrap_ci(values)
            metrics[name] = {
                'mean': mean_val,
                'median': median_val,
                'q1': q1,
                'q2': q2,
                'q3': q3,
                'q4': q4,
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'values': values  # saves raw data
            }
        else:
            metrics[name] = None
    
    return metrics

def main():
    results = {}
    for k in [5, 10, 20]:
        print(f"Computing metrics for k={k}")
        results[str(k)] = compute_metrics(k)
    
    for k in results:
        for metric_name, metric_data in results[k].items():
            if metric_data and 'values' in metric_data:
                metric_data['values'] = (
                    "[" + ", ".join(map(str, metric_data['values'])) + "]"
                )

    # Save to file
    output_path = "web_metrics_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()