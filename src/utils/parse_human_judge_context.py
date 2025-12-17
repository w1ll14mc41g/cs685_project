"""
Generate HTML context file for human evaluation of summaries.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


def _build_perspectives(points: List[str], doc_ids: List[str]) -> List[Dict]:
    """Build perspectives list from points and document IDs."""
    return [{"text": pt, "evidence_docs": [doc_ids[i]] if i < len(doc_ids) else []}
            for i, pt in enumerate(points)]


def get_gold_reference(query: str, gold_file: str = "data/theperspective/data.jsonl") -> Optional[List[Dict]]:
    """Get gold reference by matching query text."""
    try:
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("title", "").strip() == query.strip():
                    return [
                        {"claim": item.get("t1", ""), "perspectives": _build_perspectives(item.get("response1", []), item.get("favor_ids", []))},
                        {"claim": item.get("t2", ""), "perspectives": _build_perspectives(item.get("response2", []), item.get("against_ids", []))}
                    ]
    except Exception as e:
        print(f"Error loading gold data: {e}")
    return None


def _find_json_entry(query: str, file_path: str, query_key: str = "query") -> Optional[Dict]:
    """Find entry in JSON file by matching query text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for entry in json.load(f):
                if entry.get(query_key, "").strip() == query.strip():
                    return entry
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None


def get_web_docs(query: str, merged_file: str = "data/merged-corpus/merged-10.json") -> List[Dict]:
    """Extract web docs (URL entries) for a query."""
    entry = _find_json_entry(query, merged_file)
    if entry:
        return [{"id": doc.get("id"), "content": doc.get("content", "")}
               for doc in entry.get("merged", [])
               if isinstance(doc.get("id"), str) and doc.get("id", "").startswith("https://")]
    return []


def find_summary(query: str, summaries_file: str) -> Optional[Dict]:
    """Find summary entry by matching query text."""
    return _find_json_entry(query, summaries_file)


def format_claims(claims: List[Dict], title: str, css_class: str = "summary-section") -> str:
    """Format claims (summary or gold reference) as HTML."""
    html = [f'<div class="{css_class}"><h3>{title}</h3>']
    for i, claim in enumerate(claims, 1):
        html.append(f'<div class="claim"><h4>Claim {i}: {claim.get("claim", "")}</h4><ul class="perspectives">')
        for j, p in enumerate(claim.get("perspectives", []), 1):
            docs = ", ".join(str(d) for d in p.get("evidence_docs", [])) or "None"
            html.append(f'<li class="perspective"><strong>P{j}:</strong> {p.get("text", "")} <em>(docs: {docs})</em></li>')
        html.append('</ul></div>')
    html.append('</div>')
    return ''.join(html)


def format_web_docs(web_docs: List[Dict]) -> str:
    """Format web documents as HTML."""
    if not web_docs:
        return '<div class="web-docs"><h4>Relevant Web Documents</h4><p><em>No web documents available.</em></p></div>'
    html = ['<div class="web-docs"><h4>Relevant Web Documents</h4><ul>']
    for i, doc in enumerate(web_docs, 1):
        content = doc.get("content", "")
        html.append(f'<li class="web-doc"><strong>Doc {i}:</strong> <a href="{doc.get("id")}" target="_blank">{doc.get("id")}</a><div class="doc-content">{content}</div></li>')
    html.append('</ul></div>')
    return ''.join(html)


def generate_html(queries_data: List[Dict], output_path: str):
    """Generate HTML file with all evaluation context."""
    gold_file = "data/theperspective/data.jsonl"
    offline_file = "results/offline-summaries-JSON-enforced/results-10-offline-0-online-tfidf-20251214_222854.json"
    merged_file = "results/merged-summaries/results-merged-10-20251215_082353.json"
    merged_corpus_file = "data/merged-corpus/merged-10.json"
    
    html = ["""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Evaluation Context</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.3; max-width: 1600px; margin: 0 auto; padding: 0; background: #f5f5f5; font-size: 13px; padding-top: 60px; }
        .navbar { position: fixed; top: 0; left: 0; right: 0; background: #2c3e50; color: white; padding: 8px 15px; z-index: 1000; box-shadow: 0 2px 4px rgba(0,0,0,0.2); display: flex; align-items: center; gap: 15px; }
        .navbar select { padding: 5px 10px; font-size: 13px; border-radius: 3px; border: none; background: white; cursor: pointer; }
        .navbar label { font-size: 13px; font-weight: 500; }
        .query-section { background: white; padding: 12px; margin-bottom: 10px; border-radius: 3px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
        .query-hidden { display: none; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 5px; margin: 10px 0; font-size: 20px; }
        h2 { color: #555; margin: 15px 0 8px; padding: 6px 10px; background: #e8f5e9; border-left: 4px solid #4CAF50; font-size: 16px; }
        h3 { color: #666; margin: 8px 0 4px; font-size: 14px; }
        h4 { color: #777; margin: 6px 0 3px; font-size: 13px; }
        .gold-ref { margin: 8px 0; padding: 8px; background: #e3f2fd; border-left: 3px solid #2196F3; }
        .summary-section { margin: 8px 0; padding: 8px; background: #fafafa; border-radius: 2px; }
        .summary-hidden { display: none; }
        .claim { margin: 6px 0; padding: 6px; background: white; border-left: 2px solid #2196F3; }
        .perspectives { list-style: none; padding: 0; margin: 3px 0; }
        .perspective { margin: 3px 0; padding: 4px 6px; background: #f9f9f9; border-radius: 2px; }
        .web-docs { margin: 8px 0; padding: 8px; background: #fff3e0; border-radius: 2px; }
        .web-docs ul { list-style: none; padding: 0; margin: 3px 0; }
        .web-doc { margin: 4px 0; padding: 6px; background: white; border-radius: 2px; border-left: 2px solid #FF9800; }
        .doc-content { margin-top: 4px; padding: 4px; background: #fafafa; border-radius: 2px; font-size: 11px; color: #555; }
        .warning { padding: 5px 8px; background: #ffebee; border-left: 2px solid #f44336; margin: 5px 0; color: #c62828; font-size: 12px; }
        .query-ids { margin: 4px 0 8px 10px; font-size: 11px; color: #888; }
        a { color: #2196F3; word-break: break-all; }
        em { color: #888; font-size: 11px; }
    </style>
    <script>
        function navigateToQuery() {
            const select = document.getElementById('querySelect');
            const queryId = select.value;
            if (queryId) {
                document.querySelectorAll('.query-section').forEach(s => s.classList.add('query-hidden'));
                document.getElementById(queryId).classList.remove('query-hidden');
                document.getElementById(queryId).scrollIntoView({ behavior: 'smooth', block: 'start' });
                // Apply current summary type to the newly shown query
                toggleSummaryType();
            } else {
                document.querySelectorAll('.query-section').forEach(s => s.classList.remove('query-hidden'));
                toggleSummaryType();
            }
        }
        function toggleSummaryType() {
            const type = document.getElementById('summaryTypeSelect').value;
            const queryId = document.getElementById('querySelect').value;
            const sections = queryId ? [document.getElementById(queryId)] : document.querySelectorAll('.query-section');
            sections.forEach(section => {
                if (!section) return;
                const offline = section.querySelector('.offline-summary');
                const merged = section.querySelector('.merged-summary');
                const webDocs = section.querySelector('.web-docs-container');
                const isOffline = type === 'offline';
                if (offline) offline.classList.toggle('summary-hidden', !isOffline);
                if (merged) merged.classList.toggle('summary-hidden', isOffline);
                if (webDocs) webDocs.classList.toggle('summary-hidden', isOffline);
            });
        }
        // Apply summary type on page load
        window.addEventListener('DOMContentLoaded', function() {
            toggleSummaryType();
        });
    </script>
</head>
<body>
    <div class="navbar">
        <label for="querySelect">Query:</label>
        <select id="querySelect" onchange="navigateToQuery()">
            <option value="">All Queries</option>"""]
    
    for i, q in enumerate(queries_data, 1):
        html.append(f'            <option value="query-{i}">{i}. {q.get("query", "")}</option>')
    
    html.append(f"""        </select>
        <label for="summaryTypeSelect">Summary:</label>
        <select id="summaryTypeSelect" onchange="toggleSummaryType()">
            <option value="offline">Offline</option>
            <option value="merged">Merged</option>
        </select>
        <span style="margin-left: auto; font-size: 12px;">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | {len(queries_data)} queries</span>
    </div>
    <div style="padding: 10px 15px;"><h1>Human Evaluation Context</h1>""")
    
    for i, q in enumerate(queries_data, 1):
        query = q.get("query", "")
        id_offline = q.get("id_offline", "")
        id_merged = q.get("id_merged", "")
        html.append(f'<div class="query-section" id="query-{i}"><h2>{i}. {query}</h2>')
        if id_offline or id_merged:
            ids_text = []
            if id_offline:
                ids_text.append(f'ID Offline: {id_offline}')
            if id_merged:
                ids_text.append(f'ID Merged: {id_merged}')
            html.append(f'<div class="query-ids">{" | ".join(ids_text)}</div>')
        
        gold_ref = get_gold_reference(query, gold_file)
        html.append(format_claims(gold_ref, "Gold Reference", "gold-ref") if gold_ref else '<div class="warning">⚠️ Gold reference not found.</div>')
        
        offline_entry = find_summary(query, offline_file)
        offline_html = format_claims(offline_entry.get("summary", []), "Offline Summary to Evaluate") if offline_entry else '<div class="warning">⚠️ Offline summary not found.</div>'
        html.append(f'<div class="offline-summary">{offline_html}</div>')
        
        merged_entry = find_summary(query, merged_file)
        if merged_entry:
            html.append(f'<div class="merged-summary summary-hidden">{format_claims(merged_entry.get("summary", []), "Merged Summary to Evaluate")}</div>')
            html.append(f'<div class="web-docs-container summary-hidden">{format_web_docs(get_web_docs(query, merged_corpus_file))}</div>')
        else:
            html.append('<div class="merged-summary summary-hidden"><div class="warning">⚠️ Merged summary not found.</div></div>')
        
        html.append('</div>')
    
    html.append('</div></body></html>')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html))
    print(f"HTML evaluation context generated: {output_path}")


def main():
    """Main function."""
    try:
        with open("data/valid-queries/summary_eval_20.json", 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        output_path = f"data/human-eval/llm_as_judge_context-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        generate_html(queries_data, output_path)
        print(f"Successfully generated evaluation context for {len(queries_data)} queries.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
