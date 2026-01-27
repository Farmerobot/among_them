import json
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIRS = [
    os.path.join(SCRIPT_DIR, 'discussion_benchmark_results/first_dataset'),
    os.path.join(SCRIPT_DIR, 'discussion_benchmark_results/second_dataset')
]

def parse_results_file(file_path):
    """Parse a discussion benchmark results file and extract key information."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        model_name = None
        survival_rate = None
        successes = None
        total = None
        
        for line in lines[:10]:  # Header is in first 10 lines
            line = line.strip()
            
            # Extract model name
            if line.startswith('Model (target):'):
                model_name = line.replace('Model (target):', '').strip()
            
            # Extract survival rate
            if line.startswith('Survival Rate:'):
                # Format: "Survival Rate: 20.0% (2/10)"
                match = re.search(r'Survival Rate:.*?\((\d+)/(\d+)\)', line)
                if match:
                    successes = int(match.group(1))
                    total = int(match.group(2))
                    survival_rate = successes / total if total > 0 else 0.0
        
        if model_name and survival_rate is not None:
            return {
                'model': model_name,
                'survival_rate': survival_rate,
                'successes': successes,
                'total': total,
                'file': os.path.basename(file_path)
            }
    except Exception as e:
        print(f"  Error parsing {file_path}: {e}")
    
    return None

def get_stats(data_list, n_repeats=5):
    """Calculate mean and standard error from a list of values.
    
    Args:
        data_list: List of values
        n_repeats: Number of repeats (default 5) - used for SE calculation
    """
    valid_data = [d for d in data_list if d is not None]
    if len(valid_data) < 1:
        return 0, 0
    if len(valid_data) < 2:
        return valid_data[0], 0
    mean = np.mean(valid_data)
    sd = np.std(valid_data, ddof=1)
    se = sd / np.sqrt(n_repeats)
    return mean, se

def map_model_name(model_name):
    """Map model names from output to thesis names."""
    model_lower = model_name.lower()
    # Check for exact patterns first
    if 'mappo' in model_lower:
        return 'DeepSeek 1.5B MAPPO'
    elif 'sampled' in model_lower:
        return 'DeepSeek 1.5B SFT-S'
    elif 'deepseek-r1:1.5b' in model_lower or model_name == 'deepseek-r1:1.5b':
        return 'DeepSeek 1.5B (Baseline)'
    elif 'deepseek-r1:14b' in model_lower or model_name == 'deepseek-r1:14b':
        return 'DeepSeek 14B'
    elif 'deepseek/deepseek-r1' in model_name or (model_lower.startswith('deepseek') and 'r1' in model_lower and '14' not in model_lower and '1.5' not in model_lower):
        return 'DeepSeek 671B'
    elif 'among-them' in model_lower and 'gguf' in model_lower:
        return 'DeepSeek 1.5B SFT'
    return model_name  # Return original if no match

print("=" * 80)
print("PROCESSING DISCUSSION BENCHMARK RESULTS")
print("=" * 80)

# Collect all results by model
model_results = defaultdict(list)

for dataset_dir in DATASET_DIRS:
    dataset_name = os.path.basename(dataset_dir)
    print(f"\nProcessing {dataset_name}...")
    
    # Find all results files
    pattern = os.path.join(dataset_dir, '*_results.txt')
    result_files = glob(pattern)
    
    print(f"  Found {len(result_files)} result files")
    
    for file_path in sorted(result_files):
        result = parse_results_file(file_path)
        if result:
            model_name = result['model']
            model_results[model_name].append({
                'survival_rate': result['survival_rate'],
                'successes': result['successes'],
                'total': result['total'],
                'dataset': dataset_name,
                'file': result['file']
            })
            print(f"  {os.path.basename(file_path)}: {model_name} - {result['survival_rate']:.1%} ({result['successes']}/{result['total']})")

# Aggregate statistics per model
summary_data = []

print(f"\n{'=' * 80}")
print("AGGREGATING RESULTS BY MODEL")
print(f"{'=' * 80}")

N_REPEATS = 5

for model_name, results in sorted(model_results.items()):
    print(f"\nModel: {model_name}")
    
    # Separate results by dataset
    first_dataset_results = [r for r in results if r['dataset'] == 'first_dataset']
    second_dataset_results = [r for r in results if r['dataset'] == 'second_dataset']
    
    # Sort by file name to ensure consistent ordering
    first_dataset_results.sort(key=lambda x: x['file'])
    second_dataset_results.sort(key=lambda x: x['file'])
    
    print(f"  First dataset runs: {len(first_dataset_results)}")
    print(f"  Second dataset runs: {len(second_dataset_results)}")
    
    if len(first_dataset_results) != len(second_dataset_results):
        print(f"  Warning: Mismatch in number of runs between datasets!")
    
    # Combine runs by matching position (run 1 from first + run 1 from second, etc.)
    combined_runs = []
    min_runs = min(len(first_dataset_results), len(second_dataset_results))
    
    for i in range(min_runs):
        first_run = first_dataset_results[i]
        second_run = second_dataset_results[i]
        
        # Combine successes and totals
        combined_successes = first_run['successes'] + second_run['successes']
        combined_total = first_run['total'] + second_run['total']
        combined_rate = combined_successes / combined_total if combined_total > 0 else 0.0
        
        combined_runs.append({
            'run_num': i + 1,
            'successes': combined_successes,
            'total': combined_total,
            'rate': combined_rate,
            'first': first_run,
            'second': second_run
        })
    
    # Calculate statistics from combined runs
    combined_rates = [r['rate'] for r in combined_runs]
    mean, se = get_stats(combined_rates, n_repeats=N_REPEATS)
    
    # Calculate total across all combined runs
    total_successes = sum(r['successes'] for r in combined_runs)
    total_runs = sum(r['total'] for r in combined_runs)
    
    # Print per-run details
    print(f"  Combined runs (first + second dataset):")
    for combined in combined_runs:
        first = combined['first']
        second = combined['second']
        print(f"    Run {combined['run_num']:2d}: {combined['rate']:.1%} ({combined['successes']}/{combined['total']}) = "
              f"{first['successes']}/{first['total']} (first) + {second['successes']}/{second['total']} (second)")
        print(f"           First: {first['file']}")
        print(f"           Second: {second['file']}")
    
    print(f"  Summary:")
    print(f"    Overall: {total_successes}/{total_runs} = {total_successes/total_runs:.1%}")
    print(f"    Mean across {N_REPEATS} combined runs: {mean:.4f} ± SE={se:.4f}")
    print(f"    95% CI: [{mean-1.96*se:.4f}, {mean+1.96*se:.4f}]")
    
    # Map model name to thesis name
    thesis_model_name = map_model_name(model_name)
    
    summary_data.append({
        'model': thesis_model_name,
        'mean': mean,
        'se': se,
        'total_successes': total_successes,
        'total_runs': total_runs,
        'n_runs': len(combined_runs)
    })

# Print overall summary table
print(f"\n{'=' * 80}")
print("OVERALL SUMMARY - ALL MODELS")
print(f"{'=' * 80}")
print(f"{'Model':<40} {'Combined Runs':<15} {'Total':<12} {'Mean':<12} {'SE':<12} {'95% CI':<25}")
print("-" * 80)
for data in summary_data:
    ci_lower = data['mean'] - 1.96 * data['se']
    ci_upper = data['mean'] + 1.96 * data['se']
    print(f"{data['model']:<40} {data['n_runs']:<15} {data['total_successes']}/{data['total_runs']:<10} {data['mean']:<12.4f} {data['se']:<12.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

def create_plot(data, title, filename, color):
    """Create a bar plot with error bars."""
    # Sort data by mean
    sorted_data = sorted(data, key=lambda x: x['mean'])
    
    models = [d['model'] for d in sorted_data]
    means = [d['mean'] for d in sorted_data]
    ses = [d['se'] for d in sorted_data]
    
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"{'Model':<40} {'Mean':<12} {'SE':<12} {'95% CI':<25} {'Intersects Top'}")
    print("-" * 80)
    
    # Logic to detect intersections with the TOP model
    top_lower_bound = means[-1] - ses[-1]
    top_upper_bound = means[-1] + ses[-1]
    
    for i, (model, mean, se) in enumerate(zip(models, means, ses)):
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se
        intersects_top = (mean + se) >= top_lower_bound
        intersect_str = "YES" if intersects_top and i < len(models)-1 else "NO (top)" if i == len(models)-1 else "NO"
        
        print(f"{model:<40} {mean:<12.4f} {se:<12.4f} [{ci_lower:.4f}, {ci_upper:.4f}]  {intersect_str}")
    
    print(f"\nTop model: {models[-1]} (Mean={means[-1]:.4f}, Range=[{top_lower_bound:.4f}, {top_upper_bound:.4f}])")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(models))
    
    # Plot horizontal bars
    bars = ax.barh(y, means, xerr=ses, capsize=10, color=color, alpha=0.7, edgecolor='black', label='Mean ± SE')
    
    # Styling
    ax.set_title(title, fontsize=15, fontweight='bold', pad=25)
    ax.set_xlabel('Survival Rate', fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0, min(1.0, max(means) + max(ses) + 0.1))
    ax.grid(axis='x', alpha=0.2)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        w = bar.get_width()
        s = ses[i]
        
        ax.text(w + s + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.3f}', 
                va='center', fontweight='bold', color='black')
    
    plt.tight_layout()
    output_path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

# Generate plot
if summary_data:
    create_plot(summary_data, 
                'Discussion Benchmark: Survival Rate', 
                'discussion_benchmark_comparison.png', 
                '#2E86AB')
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
else:
    print("\nNo data found to plot!")
