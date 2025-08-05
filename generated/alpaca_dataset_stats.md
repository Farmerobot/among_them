# Alpaca Dataset Statistics

Tokenizer used for token counts: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

- **Training examples:** 896
- **Evaluation examples:** 224
- **Total examples:** 1,120
- **Excluded examples (assistant output exceeded 3000 tokens):** 25
  - Training excluded: 21
  - Evaluation excluded: 4

> Note: Any example whose assistant output is longer than the arbitrary cutoff of 3000 tokens is excluded from the generated dataset to keep file sizes manageable, reduce context length during training, and therefore save VRAM.

## Maximum Token Lengths Per Example
- **Longest instruction:** 13,390 tokens
- **Longest output:** 2,774 tokens
- **Longest combined (instruction + output):** 13,693 tokens

## Total Token Counts
- **Total instruction tokens:** 3,819,880 tokens
- **Total output tokens:** 515,889 tokens
- **Overall total (all instructions + all outputs):** 4,335,769 tokens

## Token Count Distribution Summary
| Statistic | Input Tokens | Output Tokens | Total Tokens |
| :-------- | :----------- | :------------ | :----------- |
| **Count** | 1,120 | 1,120 | 1,120 |
| **Min** | 897 | 80 | 1,054 |
| **Median** | 2,950.00 | 322.00 | 3,549.50 |
| **Mean** | 3,449.20 | 547.44 | 3,996.64 |
| **Max** | 13,390 | 5,748 | 13,693 |
| **Std Dev** | 2,283.09 | 676.95 | 2,279.53 |
