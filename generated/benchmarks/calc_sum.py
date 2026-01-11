import json
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. DATA PROCESSING
res_files = [
    # '20260103_125807_results', 
    ['20260103_151812_results', '20260109_233415_results'], # mappo 1
    ['20260103_171527_results', '20260109_191608_results'], # sampled 1
    ['20260103_203256_results', '20260104_174926_results'] # 1.5b 1
    # '20260103_220912_results',
    # '20260104_115959_results'
    # '20260104_174926_results', # 1.5b 2
    # '20260109_191608_results', # sampled 2
    # '20260109_233415_results', # mappo 2
]

N_REPEATS = 10
summary_data = []

def get_stats(data_list):
    valid_data = [d for d in data_list if d > 0]
    if len(valid_data) < 2: return 0, 0
    mean = np.mean(valid_data)
    se = np.std(valid_data, ddof=1) / np.sqrt(len(valid_data))
    return mean, se

for res_file_entry in res_files:
    # Normalize to list: handle both string and list inputs
    if isinstance(res_file_entry, str):
        file_list = [res_file_entry]
    else:
        file_list = res_file_entry
    
    # Collect all runs from all files
    all_runs_by_example = {}  # example_index -> list of all runs across files
    
    # First, determine number of examples from first valid file
    examples = None
    model_name = None
    
    for res_file in file_list:
        file_path = res_file + '.json'
        if not os.path.exists(file_path):
            continue
        
        with open(file_path, 'r') as f:
            res = json.load(f)
        
        # Set examples count from first file
        if examples is None:
            examples = len(res['examples'])
        
        # Get model name from first available file
        if model_name is None:
            model_name = res_file
            txt_path = res_file + '.txt'
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    model_name = f.readline().replace('Model:', '').strip()
        
        # Collect runs for each example
        for example_idx, example in enumerate(res['examples']):
            if example_idx not in all_runs_by_example:
                all_runs_by_example[example_idx] = []
            
            # Add all runs from this file for this example
            all_runs_by_example[example_idx].extend(example['runs'])
    
    # If no valid files found, skip
    if examples is None:
        continue
    
    # Now aggregate scores across all runs
    # We'll re-index runs sequentially: 1, 2, 3, ..., N_REPEATS
    total_scores = [0] * N_REPEATS
    total_scores_succ = [0] * N_REPEATS
    count_succ = [0] * N_REPEATS
    
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
    
    summary_data.append({
        'model': model_name,
        'mean_total': mean_t, 'se_total': se_t,
        'mean_succ': mean_s, 'se_succ': se_s
    })

def create_plot(data, key_mean, key_se, title, filename, color):
    # Sort data for this specific plot
    sorted_data = sorted(data, key=lambda x: x[key_mean])
    
    models = [d['model'] for d in sorted_data]
    means = [d[key_mean] for d in sorted_data]
    ses = [d[key_se] for d in sorted_data]
    
    fig, ax = plt.subplots(figsize=(14, 9))
    x = np.arange(len(models))
    
    # Plot Bars
    bars = ax.bar(x, means, yerr=ses, capsize=10, color=color, alpha=0.7, edgecolor='black', label='Mean ± SE')
    
    # Add ALL error boundary lines
    for i in range(len(models)):
        upper = means[i] + ses[i]
        lower = means[i] - ses[i]
        # Draw horizontal lines across the whole plot
        ax.axhline(y=upper, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
        ax.axhline(y=lower, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    # Styling
    ax.set_title(title, fontsize=15, fontweight='bold', pad=25)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha='right')
    ax.set_ylim(0, max(means) + max(ses) + 0.1)
    ax.grid(axis='y', alpha=0.2)

    # Logic to detect intersections with the TOP model
    top_lower_bound = means[-1] - ses[-1]
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        s = ses[i]
        intersects_top = (h + s) >= top_lower_bound
        label_color = 'red' if intersects_top and i < len(models)-1 else 'black'
        
        ax.text(bar.get_x() + bar.get_width()/2, h + s + 0.01, f'{h:.3f}', 
                ha='center', fontweight='bold', color=label_color)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

# Generate the two requested plots
create_plot(summary_data, 'mean_total', 'se_total', 
            'Comparison 1: Total Performance (All Runs)', 
            'plot_total_performance.png', '#2E86AB')

create_plot(summary_data, 'mean_succ', 'se_succ', 
            'Comparison 2: Successful Runs Only (Quality Analysis)', 
            'plot_successful_performance.png', '#A23B72')