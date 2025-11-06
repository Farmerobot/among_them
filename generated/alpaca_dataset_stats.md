# Alpaca Dataset Statistics

Tokenizer used for token counts: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

- **Training conversations:** 367
- **Evaluation conversations:** 92
- **Total conversations:** 459

> **Soft Mode (Default):** Conversations are split at 3000 tokens. If a single user-assistant pair exceeds this limit, it becomes its own chunk (preserves all data). Maximum chunk size may exceed limit.

## Maximum Token Lengths
- **Longest conversation input (all user turns):** 1,833 tokens
- **Longest conversation output (all assistant turns):** 6,442 tokens
- **Longest single turn output:** 6,442 tokens
- **Longest combined (instruction + output):** 6,634 tokens

## Total Token Counts
- **Total input tokens (all user turns):** 345,171 tokens
- **Total output tokens (all assistant turns):** 694,314 tokens
- **Overall total (all inputs + all outputs):** 1,039,485 tokens

## Token Count Distribution Summary
| Statistic | Input Tokens | Output Tokens | Total Tokens |
| :-------- | :----------- | :------------ | :----------- |
| **Count** | 459 | 459 | 459 |
| **Min** | 39 | 71 | 226 |
| **Median** | 686 | 1,350 | 2,391 |
| **Mean** | 752.01 | 1,512.67 | 2,264.67 |
| **Max** | 1,833 | 6,442 | 6,634 |
| **Std Dev** | 473.44 | 1,047.99 | 1,013.65 |
