# Sampled Alpaca Dataset Statistics

The script generating this stats hasn't performed any token counting (token counts are taken from the original files).

- **Total examples:** 1039 (out of original 1120 - 92.8% preserved)

## Trace Evaluation Distribution
| Dataset | Eval 0 | Eval 1 | Eval 2 | Eval 3 |
| :-------- | :----------- | :------------ | :----------- | :----------- |
| **Original** | 33 | 46 | 327 | 712 |
| **Sampled** | 0 | 0 | 327 | 712 |

## Maximum Token Lengths Per Example
- **Longest instruction:** 13,390 tokens
- **Longest output:** 5,089 tokens
- **Longest combined (instruction + output):** 13,693 tokens

## Total Token Counts
- **Total instruction tokens:** 3,616,441 tokens
- **Total output tokens:** 555,401 tokens
- **Overall total (all instructions + all outputs):** 4,171,842 tokens

## Token Count Distribution Summary
| Statistic | Input Tokens | Output Tokens | Total Tokens |
| :-------- | :----------- | :------------ | :----------- |
| **Count** | 1,039 | 1,039 | 1,039 |
| **Min** | 897 | 80 | 1,054 |
| **Median** | 2,995.00 | 318.00 | 3,561.00 |
| **Mean** | 3,480.69 | 534.55 | 4,015.25 |
| **Max** | 13,390 | 5,089 | 13,693 |
| **Std Dev** | 2,322.43 | 636.74 | 2,305.61 |
