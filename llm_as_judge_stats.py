'''
This file should read from results/evaluation/merged/ and results/evaluation/offline,
it should calculate the mean, median, q1, q2, q3, q4, and a bootstrapped 95% confidence interval for each json grouped by k.
'''

import json
import os
import numpy as np

def bootstrap_ci(data, n_boot=1000, ci=95):
    """
    Compute bootstrapped confidence interval for the mean.
    """
    if len(data) == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper

def main():
    base_dir = os.path.dirname(__file__)
    dirs = {
        'merged': os.path.join(base_dir, 'results', 'evaluation', 'merged'),
        'offline': os.path.join(base_dir, 'results', 'evaluation', 'offline')
    }
    
    results_stats = {}
    
    for group, dir_path in dirs.items():
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            continue
        results_stats[group] = {}
        files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        for file in files:
            # Extract k from filename, e.g., merged_5_llm_judge_scores_...
            parts = file.split('_')
            k = int(parts[1])
            
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            total_scores = []
            
            for res in results:
                raw_response = res.get('raw_response', '')
                if raw_response:
                    try:
                        response_data = json.loads(raw_response)
                        res_scores = response_data.get('scores', {})
                        # Compute total_score as sum of all criteria
                        score_sum = 0
                        count = 0
                        for key, value in res_scores.items():
                            if key.startswith('criterion_'):
                                score_sum += value
                                count += 1
                        if count == 5:  # Ensure all 5 criteria are present
                            total_scores.append(score_sum)
                    except json.JSONDecodeError:
                        pass  # Skip if can't parse
            
            results_stats[group][k] = {}
            # Compute statistics for total_score
            if not total_scores:
                results_stats[group][k]['total_score'] = {'error': 'No data'}
            else:
                vals = np.array(total_scores, dtype=float)
                mean_val = np.mean(vals)
                median_val = np.median(vals)
                q1 = np.percentile(vals, 25)
                q2 = np.percentile(vals, 50)  # median
                q3 = np.percentile(vals, 75)
                q4 = np.percentile(vals, 100)  # max
                ci_lower, ci_upper = bootstrap_ci(vals)
                
                results_stats[group][k]['total_score'] = {
                    'mean': round(mean_val, 3),
                    'median': round(median_val, 3),
                    'q1': round(q1, 3),
                    'q2': round(q2, 3),
                    'q3': round(q3, 3),
                    'q4': round(q4, 3),
                    'ci_lower': round(ci_lower, 3),
                    'ci_upper': round(ci_upper, 3)
                }
    
    # Save to JSON
    output_path = os.path.join(base_dir, 'results', 'llm_judge_stats.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_stats, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()