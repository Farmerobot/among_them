import json
import numpy as np
import matplotlib.pyplot as plt

# Data: survival counts for each model across 5 runs
# Out of 50 situations where "deepseek/deepseek-r1" was initially ejected
model_data = {
    'deepseek/deepseek-r1': [5, 7, 4, 7, 5],
    'hf.co/Luncenok/deepseek-r1-among-them-gguf': [6, 3, 4, 3, 5],
    'deepseek-r1:1.5b': [4, 2, 4, 5, 4],
    'hf.co/Luncenok/deepseek-r1-among-them-gguf-sampled': [5, 4, 5, 5, 4],
    'hf.co/Luncenok/deepseek-r1-among-them-mappo-gguf': [3, 6, 5, 5, 5],
}

# Calculate statistics for each model
summary_data = []
for model_name, counts in model_data.items():
    counts_array = np.array(counts)
    mean = np.mean(counts_array)
    std = np.std(counts_array, ddof=1)  # Sample standard deviation
    total = np.sum(counts_array)
    
    summary_data.append({
        'model': model_name,
        'counts': counts,
        'mean': round(mean, 2),
        'std': round(std, 2),
        'total': int(total),
        'lower_bound': round(mean - std, 2),
        'upper_bound': round(mean + std, 2)
    })
    
    print(f"{model_name}:")
    print(f"  Counts: {counts}")
    print(f"  Mean: {mean:.2f} ± {std:.2f}")
    print(f"  Total: {total}")
    print(f"  Range: [{mean - std:.2f} - {mean + std:.2f}]")
    print()

# Sort by mean (best to worst)
summary_data.sort(key=lambda x: x['mean'], reverse=True)

# Prepare data for plotting
models = [item['model'] for item in summary_data]
means = [item['mean'] for item in summary_data]
stds = [item['std'] for item in summary_data]
totals = [item['total'] for item in summary_data]

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
y_pos = np.arange(len(models))
bars = ax.barh(y_pos, means, xerr=stds, capsize=5, alpha=0.7, 
               color=['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c'])

# Add value labels on bars
for i, (mean, std, total) in enumerate(zip(means, stds, totals)):
    ax.text(mean + std + 0.1, i, f'{mean:.2f} ± {std:.2f}\n(total: {total})', 
            va='center', fontsize=9)

# Customize the plot
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=10)
ax.set_xlabel('Mean Survival Count (out of 10 per run)', fontsize=11, fontweight='bold')
ax.set_title('Discussion Benchmark: Survival Rates\n(Ordered from best to worst)', 
             fontsize=12, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(left=0, right=max([m + s for m, s in zip(means, stds)]) * 1.3)

# Add vertical line at x=5 (50% survival rate)
ax.axvline(x=5, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='50% baseline')
ax.legend(loc='lower right')

plt.tight_layout()

# Save the plot
output_file = 'discussion_survival_chart.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nChart saved to: {output_file}")

# Save summary data to JSON
json_output = 'discussion_survival_summary.json'
with open(json_output, 'w') as f:
    json.dump(summary_data, f, indent=2)
print(f"Summary data saved to: {json_output}")

# Print ordered summary
print("\n" + "="*80)
print("ORDERED SUMMARY (Best to Worst):")
print("="*80)
for i, item in enumerate(summary_data, 1):
    print(f"{i}. {item['model']}")
    print(f"   Mean: {item['mean']:.2f} ± {item['std']:.2f} (Total: {item['total']}/50)")
    print()

plt.show()

