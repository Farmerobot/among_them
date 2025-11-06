# Alpaca Dataset Statistics

Tokenizer used for token counts: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

- **Training conversations:** 328
- **Evaluation conversations:** 83
- **Total conversations:** 411

> **Strict Mode:** Conversations are split at 3000 tokens. User-assistant pairs exceeding this limit individually are **discarded** (data loss). This creates a hard limit to prevent OOM.

## Maximum Token Lengths
- **Longest conversation input (all user turns):** 1,833 tokens
- **Longest conversation output (all assistant turns):** 2,850 tokens
- **Longest single turn output:** 2,850 tokens
- **Longest combined (instruction + output):** 3,000 tokens

## Total Token Counts
- **Total input tokens (all user turns):** 324,365 tokens
- **Total output tokens (all assistant turns):** 517,021 tokens
- **Overall total (all inputs + all outputs):** 841,386 tokens

## Token Count Distribution Summary
| Statistic | Input Tokens | Output Tokens | Total Tokens |
| :-------- | :----------- | :------------ | :----------- |
| **Count** | 411 | 411 | 411 |
| **Min** | 39 | 71 | 226 |
| **Median** | 743 | 1,261 | 2,213 |
| **Mean** | 789.21 | 1,257.96 | 2,047.17 |
| **Max** | 1,833 | 2,850 | 3,000 |
| **Std Dev** | 469.17 | 705.24 | 777.47 |
