# Sampled Alpaca Dataset Statistics

The script generating this stats hasn't performed any token counting (token counts are taken from the original files).

- **Total examples:** 493 (out of original 573 - 86.0% preserved)

## Trace Evaluation Distribution
| Dataset | Eval 0 | Eval 1 | Eval 2 | Eval 3 |
| :-------- | :----------- | :------------ | :----------- | :----------- |
| **Original** | 39 | 41 | 171 | 322 |
| **Sampled** | 0 | 0 | 171 | 322 |

## Maximum Token Lengths Per Example
- **Longest instruction:** 7,722 tokens
- **Longest output:** 1,814 tokens
- **Longest combined (instruction + output):** 7,981 tokens

## Total Token Counts
- **Total instruction tokens:** 1,217,416 tokens
- **Total output tokens:** 186,844 tokens
- **Overall total (all instructions + all outputs):** 1,404,260 tokens

## Token Count Distribution Summary
| Statistic | Input Tokens | Output Tokens | Total Tokens |
| :-------- | :----------- | :------------ | :----------- |
| **Count** | 493 | 493 | 493 |
| **Min** | 315 | 87 | 569 |
| **Median** | 2,271.00 | 329.00 | 2,634.00 |
| **Mean** | 2,469.40 | 378.99 | 2,848.40 |
| **Max** | 7,722 | 1,814 | 7,981 |
| **Std Dev** | 1,584.37 | 206.47 | 1,583.87 |
