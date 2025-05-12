"""
Visualization script for retrieval evaluation results

This script creates visualizations of retrieval evaluation results:
- Bar charts comparing metrics across retrieval systems
- Line charts showing precision-recall curves
- Radar charts visualizing multi-dimensional performance
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

def load_results(results_file):
    """Load evaluation results from file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_metrics_bars(results, output_dir):
    """Create bar charts comparing metrics across systems"""
    plt.figure(figsize=(15, 10))
    
    systems = list(results.keys())
    map_scores = [results[system]['map'] for system in systems]
    mrr_scores = [results[system]['mrr'] for system in systems]  # Added MRR
    
    # Plot MAP
    plt.subplot(3, 2, 1)
    bars = plt.bar(systems, map_scores)
    plt.title('Mean Average Precision (MAP)')
    plt.ylabel('MAP')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # Plot MRR
    plt.subplot(3, 2, 2)
    bars = plt.bar(systems, mrr_scores)
    plt.title('Mean Reciprocal Rank (MRR)')
    plt.ylabel('MRR')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # Create bar chart for Precision@10
    prec_10 = [results[system]['precision_at_k']['10'] for system in systems]
    
    plt.subplot(3, 2, 3)
    bars = plt.bar(systems, prec_10)
    plt.title('Precision@10')
    plt.ylabel('Precision')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # Create bar chart for NDCG@10
    ndcg_10 = [results[system]['ndcg_at_k']['10'] for system in systems]
    
    plt.subplot(3, 2, 4)
    bars = plt.bar(systems, ndcg_10)
    plt.title('NDCG@10')
    plt.ylabel('NDCG')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # Create bar chart for Recall@10
    recall_10 = [results[system]['recall_at_k']['10'] for system in systems]
    
    plt.subplot(3, 2, 5)
    bars = plt.bar(systems, recall_10)
    plt.title('Recall@10')
    plt.ylabel('Recall')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    # Create bar chart for retrieval time
    times = [results[system]['average_retrieval_time'] for system in systems]
    
    plt.subplot(3, 2, 6)
    bars = plt.bar(systems, times)
    plt.title('Retrieval Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()

def plot_precision_recall_curve(results, output_dir):
    """Create line charts showing precision-recall curves"""
    plt.figure(figsize=(12, 8))
    
    # Precision at different k values
    # Assuming all systems have the same k_values keys in 'precision_at_k'
    k_values = sorted([int(k) for k in results[list(results.keys())[0]]['precision_at_k'].keys()])
    
    for system in results:
        precision_values = [results[system]['precision_at_k'][str(k)] for k in k_values]
        recall_values = [results[system]['recall_at_k'][str(k)] for k in k_values]
        
        plt.plot(recall_values, precision_values, 'o-', label=system)
    
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()


def plot_metrics_by_k(results, output_dir):
    """Create line charts showing metrics at different k values"""
    plt.figure(figsize=(15, 10))
    
    # Get k values
    k_values = sorted([int(k) for k in results[list(results.keys())[0]]['precision_at_k'].keys()])
    
    # Plot precision@k
    plt.subplot(2, 2, 1)
    for system in results:
        precision_values = [results[system]['precision_at_k'][str(k)] for k in k_values]
        plt.plot(k_values, precision_values, 'o-', label=system)
    
    plt.title('Precision@k')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    # Plot NDCG@k
    plt.subplot(2, 2, 2)
    for system in results:
        ndcg_values = [results[system]['ndcg_at_k'][str(k)] for k in k_values]
        plt.plot(k_values, ndcg_values, 'o-', label=system)
    
    plt.title('NDCG@k')
    plt.xlabel('k')
    plt.ylabel('NDCG')
    plt.legend()
    plt.grid(True)
    
    # Plot recall@k
    plt.subplot(2, 2, 3)
    for system in results:
        recall_values = [results[system]['recall_at_k'][str(k)] for k in k_values]
        plt.plot(k_values, recall_values, 'o-', label=system)
    
    plt.title('Recall@k')
    plt.xlabel('k')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    # Plot F1@k
    plt.subplot(2, 2, 4)
    for system in results:
        f1_values = [results[system]['f1_score_at_k'][str(k)] for k in k_values]
        plt.plot(k_values, f1_values, 'o-', label=system)
    
    plt.title('F1@k')
    plt.xlabel('k')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_k.png'))
    plt.close()
    
    # Create a separate figure for MAP, MRR and Time summary
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    
    text = "Mean Average Precision (MAP):\n"
    for system in results:
        text += f"{system}: {results[system]['map']:.4f}\n"
    
    text += "\nMean Reciprocal Rank (MRR):\n"
    for system in results:
        text += f"{system}: {results[system]['mrr']:.4f}\n"
    
    text += "\nRetrieval Time (seconds):\n"
    for system in results:
        text += f"{system}: {results[system]['average_retrieval_time']:.4f}\n"
    
    plt.text(0.1, 0.5, text, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_mrr_time_summary.png'))
    plt.close()


def plot_radar_chart(results, output_dir):
    """Create radar charts visualizing multi-dimensional performance"""
    systems = list(results.keys())
    # Include NDCG and MRR in the radar chart
    metrics = ['MAP', 'MRR', 'NDCG@10', 'P@10', 'R@10', 'F1@10', 'Speed']
    
    plt.figure(figsize=(10, 10))
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], metrics)
    
    # Calculate max_values for normalization
    max_val_map = max(results[s]['map'] for s in systems)
    max_val_mrr = max(results[s]['mrr'] for s in systems)
    max_val_ndcg10 = max(results[s]['ndcg_at_k']['10'] for s in systems)
    max_val_p10 = max(results[s]['precision_at_k']['10'] for s in systems)
    max_val_r10 = max(results[s]['recall_at_k']['10'] for s in systems)
    max_val_f10 = max(results[s]['f1_score_at_k']['10'] for s in systems)
    max_val_speed = max((1 / results[s]['average_retrieval_time']) if results[s]['average_retrieval_time'] > 0 else 0 for s in systems)

    max_values_norm = [max_val_map, max_val_mrr, max_val_ndcg10, max_val_p10, max_val_r10, max_val_f10, max_val_speed]

    for system in systems:
        avg_time = results[system]['average_retrieval_time']
        speed = (1 / avg_time) if avg_time > 0 else 0
        
        values = [
            results[system]['map'],
            results[system]['mrr'],
            results[system]['ndcg_at_k']['10'],
            results[system]['precision_at_k']['10'],
            results[system]['recall_at_k']['10'],
            results[system]['f1_score_at_k']['10'],
            speed
        ]
        
        normalized_values = [(v / max_v) if max_v > 0 else 0 for v, max_v in zip(values, max_values_norm)]
        normalized_values += normalized_values[:1]
        
        ax.plot(angles, normalized_values, linewidth=2, label=system)
        ax.fill(angles, normalized_values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Radar Chart of Retrieval System Performance')
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'))
    plt.close()

def create_performance_table(results, output_dir):
    """Create a table summarizing performance metrics"""
    systems = list(results.keys())
    # Assuming all systems have the same k_values keys in 'precision_at_k'
    k_values = sorted([int(k) for k in results[systems[0]]['precision_at_k'].keys()])
    
    dfs = []
    
    # Add MAP and MRR
    map_data = {'System': systems, 'MAP': [results[s]['map'] for s in systems]}
    dfs.append(pd.DataFrame(map_data))
    
    mrr_data = {'System': systems, 'MRR': [results[s]['mrr'] for s in systems]}
    dfs.append(pd.DataFrame(mrr_data))
    
    # Add NDCG@k
    for k in k_values:
        ndcg_data = {'System': systems, f'NDCG@{k}': [results[s]['ndcg_at_k'][str(k)] for s in systems]}
        dfs.append(pd.DataFrame(ndcg_data))
    
    for k in k_values:
        prec_data = {'System': systems, f'Precision@{k}': [results[s]['precision_at_k'][str(k)] for s in systems]}
        dfs.append(pd.DataFrame(prec_data))
    
    for k in k_values:
        recall_data = {'System': systems, f'Recall@{k}': [results[s]['recall_at_k'][str(k)] for s in systems]}
        dfs.append(pd.DataFrame(recall_data))
    
    for k in k_values:
        f1_data = {'System': systems, f'F1@{k}': [results[s]['f1_score_at_k'][str(k)] for s in systems]}
        dfs.append(pd.DataFrame(f1_data))
    
    time_data = {'System': systems, 'Retrieval Time': [results[s]['average_retrieval_time'] for s in systems]}
    dfs.append(pd.DataFrame(time_data))
    
    summary_df = dfs[0]
    for df in dfs[1:]:
        summary_df = pd.merge(summary_df, df, on='System')
    
    summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    styled_df = summary_df.style.format({col: '{:.4f}' for col in summary_df.columns if col != 'System'})
    html_table = styled_df.to_html()
    
    with open(os.path.join(output_dir, 'performance_summary.html'), 'w') as f:
        f.write(html_table)
    
    return summary_df

def main(results_file=None):
    """Generate visualizations for retrieval evaluation results"""
    # If no results file is provided, use the most recent one
    if results_file is None:
        # Get the most recent results file
        results_dir = Path('evaluation_results')
        if not results_dir.exists():
            print("No evaluation results directory found")
            return
        
        result_files = list(results_dir.glob('retrieval_eval_results_*.json'))
        if not result_files:
            print("No evaluation results files found")
            return
        
        results_file = str(sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])
    
    # Load results
    results = load_results(results_file)
    
    # Create output directory for visualizations
    output_dir = 'evaluation_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print(f"Generating visualizations for {results_file}...")
    plot_metrics_bars(results, output_dir)
    plot_precision_recall_curve(results, output_dir)
    plot_metrics_by_k(results, output_dir)
    plot_radar_chart(results, output_dir)
    
    # Create performance summary table
    summary_df = create_performance_table(results, output_dir)
    print(f"Performance summary:\n{summary_df}")
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
