import json
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. DATA PROCESSING
res_files = [
    ['20260103_125807_results', '20260109_212333_results'], # gguf
    ['20260103_151812_results', '20260109_233415_results'], # mappo
    ['20260103_171527_results', '20260109_191608_results'], # sampled
    ['20260103_203256_results', '20260104_174926_results'], # 1.5b
    ['20260103_220912_results', '20260110_104719_results'], # r1
    ['20260104_115959_results', '20260110_181921_results'], # 14b
    # '20260103_125807_results', # gguf 1
    # '20260103_220912_results', # r1 1
    # '20260104_115959_results' # 14b 1
    # '20260104_174926_results', # 1.5b 2
    # '20260109_191608_results', # sampled 2
    # '20260109_233415_results', # mappo 2
    # '20260109_212333_results', # gguf 2
    # '20260110_104719_results', # r1 2
    # '20260110_114425_results', # gguf 3
    # '20260110_181921_results', # 14b 2
]

N_REPEATS = 10
summary_data = []

def get_stats(data_list):
    valid_data = [d for d in data_list if d > 0]
    if len(valid_data) < 2: return 0, 0
    mean = np.mean(valid_data)
    se = np.std(valid_data, ddof=1) / np.sqrt(len(valid_data))
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
print("PROCESSING BENCHMARK RESULTS")
print("=" * 80)

for res_file_entry in res_files:
    # Normalize to list: handle both string and list inputs
    if isinstance(res_file_entry, str):
        file_list = [res_file_entry]
    else:
        file_list = res_file_entry
    
    print(f"\nProcessing files: {', '.join(file_list)}")
    
    # Collect all runs from all files
    all_runs_by_example = {}  # example_index -> list of all runs across files
    
    # First, determine number of examples from first valid file
    examples = None
    model_name = None
    total_runs_collected = 0
    
    for res_file in file_list:
        file_path = "benchmark_results/" + res_file + '.json'
        if not os.path.exists(file_path):
            print(f"  Warning: File not found: {file_path}")
            continue
        
        with open(file_path, 'r') as f:
            res = json.load(f)
        
        # Set examples count from first file
        if examples is None:
            examples = len(res['examples'])
            print(f"  Found {examples} examples")
        
        # Get model name from first available file
        if model_name is None:
            model_name = res_file
            # Try both with and without benchmark_results/ prefix
            txt_path = "benchmark_results/" + res_file + '.txt'
            if not os.path.exists(txt_path):
                txt_path = res_file + '.txt'
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    model_name = f.readline().replace('Model:', '').strip()
                    if not model_name:  # If empty, fall back to filename
                        model_name = res_file
            print(f"  Model: {model_name}")
        
        # Collect runs for each example
        runs_in_file = 0
        for example_idx, example in enumerate(res['examples']):
            if example_idx not in all_runs_by_example:
                all_runs_by_example[example_idx] = []
            
            # Add all runs from this file for this example
            all_runs_by_example[example_idx].extend(example['runs'])
            runs_in_file += len(example['runs'])
        
        total_runs_collected += runs_in_file
        print(f"  Collected {runs_in_file} runs from {res_file}")
    
    # If no valid files found, skip
    if examples is None:
        print("  Skipping: No valid files found")
        continue
    
    print(f"  Total runs collected: {total_runs_collected}")
    
    # Now aggregate scores across all runs
    # We'll re-index runs sequentially: 1, 2, 3, ..., N_REPEATS
    total_scores = [0] * N_REPEATS
    total_scores_succ = [0] * N_REPEATS
    count_succ = [0] * N_REPEATS
    count_ideal = [0] * N_REPEATS  # Track is_ideal counts
    
    for example_idx in range(examples):
        if example_idx not in all_runs_by_example:
            continue
        
        runs = all_runs_by_example[example_idx]
        
        # Re-index runs sequentially (1 to N_REPEATS)
        for new_idx, run in enumerate(runs[:N_REPEATS]):  # Limit to N_REPEATS
            total_scores[new_idx] += run['score']
            if run['success']:
                total_scores_succ[new_idx] += run['score']
                count_succ[new_idx] += 1
                # Track is_ideal (only for successful runs)
                if run.get('is_ideal', False):
                    count_ideal[new_idx] += 1
    
    # Normalize
    for i in range(N_REPEATS):
        total_scores[i] /= examples
        total_scores_succ[i] = total_scores_succ[i] / count_succ[i] if count_succ[i] > 0 else 0
    
    # Calculate stats
    mean_t, se_t = get_stats(total_scores)
    mean_s, se_s = get_stats(total_scores_succ)
    
    # Use combined model name (or first file name if no model name found)
    if model_name is None and file_list:
        model_name = file_list[0]
        print(f"  Model: {model_name}")
    
    # Calculate overall averages
    total_runs = examples * N_REPEATS
    total_successful = sum(count_succ)
    total_ideal = sum(count_ideal)
    avg_success_rate = (total_successful / total_runs * 100) if total_runs > 0 else 0
    avg_ideal_rate = (total_ideal / total_successful * 100) if total_successful > 0 else 0
    
    # Print per-run statistics
    print(f"\n  Per-run average scores for {model_name} (across {examples} examples):")
    for i in range(N_REPEATS):
        success_rate = (count_succ[i] / examples * 100) if examples > 0 else 0
        ideal_rate = (count_ideal[i] / count_succ[i] * 100) if count_succ[i] > 0 else 0
        print(f"    Run {i+1:2d}: Total={total_scores[i]:.4f}, Successful={total_scores_succ[i]:.4f} (success rate: {success_rate:.1f}%, direct parse rate: {ideal_rate:.1f}%)")
    
    print(f"  Summary statistics for {model_name}:")
    print(f"    Total (all runs):     Mean={mean_t:.4f} ± SE={se_t:.4f} (95% CI: [{mean_t-1.96*se_t:.4f}, {mean_t+1.96*se_t:.4f}])")
    print(f"    Successful runs only: Mean={mean_s:.4f} ± SE={se_s:.4f} (95% CI: [{mean_s-1.96*se_s:.4f}, {mean_s+1.96*se_s:.4f}])")
    print(f"    Average success rate: {avg_success_rate:.1f}%")
    print(f"    Average direct parse rate:   {avg_ideal_rate:.1f}%")
    
    # Map model name to thesis name
    thesis_model_name = map_model_name(model_name)
    
    summary_data.append({
        'model': thesis_model_name,
        'mean_total': mean_t, 'se_total': se_t,
        'mean_succ': mean_s, 'se_succ': se_s,
        'avg_success_rate': avg_success_rate,
        'avg_ideal_rate': avg_ideal_rate
    })

def create_plot(data, key_mean, key_se, title, filename, color):
    # Sort data for this specific plot
    sorted_data = sorted(data, key=lambda x: x[key_mean])
    
    models = [d['model'] for d in sorted_data]
    means = [d[key_mean] for d in sorted_data]
    ses = [d[key_se] for d in sorted_data]
    
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
    ax.set_xlabel('Accuracy Score', fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0, max(means) + max(ses) + 0.1)
    ax.grid(axis='x', alpha=0.2)
    
    for i, bar in enumerate(bars):
        w = bar.get_width()
        s = ses[i]
        intersects_top = (w + s) >= top_lower_bound
        label_color = 'red' if intersects_top and i < len(models)-1 else 'black'
        
        ax.text(w + s + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.3f}', 
                va='center', fontweight='bold', color=label_color)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

def create_combined_plot(data, title, filename):
    # Sort data by total mean for consistent ordering
    sorted_data = sorted(data, key=lambda x: x['mean_total'])
    
    models = [d['model'] for d in sorted_data]
    means_total = [d['mean_total'] for d in sorted_data]
    ses_total = [d['se_total'] for d in sorted_data]
    means_succ = [d['mean_succ'] for d in sorted_data]
    ses_succ = [d['se_succ'] for d in sorted_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(models))
    height = 0.35  # Height of bars
    
    # Plot horizontal bars side by side - Total Mean on top (first in legend)
    bars1 = ax.barh(y + height/2, means_total, height, xerr=ses_total, capsize=5, 
                   color='#2E86AB', alpha=0.7, edgecolor='black', label='Total Mean')
    bars2 = ax.barh(y - height/2, means_succ, height, xerr=ses_succ, capsize=5,
                   color='#A23B72', alpha=0.7, edgecolor='black', label='Successful Mean')
    
    # Styling
    ax.set_title(title, fontsize=15, fontweight='bold', pad=25)
    ax.set_xlabel('Accuracy Score', fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0, max(max(means_total) + max(ses_total), max(means_succ) + max(ses_succ)) + 0.1)
    ax.grid(axis='x', alpha=0.2)
    ax.legend(fontsize=11)
    
    # Add value labels on bars
    for bars, means, ses in [(bars1, means_total, ses_total), (bars2, means_succ, ses_succ)]:
        for i, bar in enumerate(bars):
            w = bar.get_width()
            s = ses[i]
            ax.text(w + s + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.3f}', 
                    va='center', fontweight='bold', color='black', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

# Print overall summary
print(f"\n{'=' * 100}")
print("OVERALL SUMMARY - ALL MODELS")
print(f"{'=' * 100}")
# Find max model name length for dynamic column width
max_model_len = max(len(d['model']) for d in summary_data) if summary_data else 40
model_col_width = max(50, max_model_len + 5)
print(f"{'Model Name':<{model_col_width}} {'Total Mean':<15} {'Total SE':<15} {'Succ Mean':<15} {'Succ SE':<15} {'Success Rate':<15} {'Direct Parse Rate':<15}")
print("-" * (model_col_width + 90))
for data in summary_data:
    print(f"{data['model']:<{model_col_width}} {data['mean_total']:<15.4f} {data['se_total']:<15.4f} {data['mean_succ']:<15.4f} {data['se_succ']:<15.4f} {data['avg_success_rate']:<15.1f} {data['avg_ideal_rate']:<15.1f}")

# Generate the plots
create_plot(summary_data, 'mean_total', 'se_total', 
            'Action Selection Benchmark: Total Mean', 
            'plot_total_performance.png', '#2E86AB')

create_plot(summary_data, 'mean_succ', 'se_succ', 
            'Action Selection Benchmark: Successful Mean', 
            'plot_successful_performance.png', '#A23B72')

create_combined_plot(summary_data,
                     'Action Selection Benchmark: Total Mean vs Successful Mean',
                     'plot_combined_performance.png')

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")