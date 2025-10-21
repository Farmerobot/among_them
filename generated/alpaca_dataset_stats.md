# Alpaca Dataset Statistics

Tokenizer used for token counts: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

- **Training conversations:** 100
- **Evaluation conversations:** 28
- **Total conversations:** 128
- **Excluded examples (assistant output exceeded 3000 tokens):** 44
  - Training excluded: 37
  - Evaluation excluded: 7

> Note: Any example whose assistant output is longer than the arbitrary cutoff of 3000 tokens is excluded from the generated dataset to keep file sizes manageable, reduce context length during training, and therefore save VRAM.

## Maximum Token Lengths
- **Longest conversation input (all user turns):** 3,230 tokens
- **Longest conversation output (all assistant turns):** 9,481 tokens
- **Longest single turn output:** 2,984 tokens
- **Longest combined (instruction + output):** 15,634 tokens

## Total Token Counts
- **Total input tokens (all user turns):** 247,931 tokens
- **Total output tokens (all assistant turns):** 388,017 tokens
- **Overall total (all inputs + all outputs):** 635,948 tokens

## Token Count Distribution Summary
| Statistic | Input Tokens | Output Tokens | Total Tokens |
| :-------- | :----------- | :------------ | :----------- |
| **Count** | 172 | 172 | 172 |
| **Min** | 1,040 | 192 | 1,257 |
| **Median** | 1,981.50 | 3,659.00 | 5,746.50 |
| **Mean** | 2,006.81 | 4,036.71 | 6,043.52 |
| **Max** | 3,396 | 12,524 | 15,634 |
| **Std Dev** | 589.88 | 2,643.07 | 3,070.92 |
