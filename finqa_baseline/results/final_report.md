# Final Baseline Report (FinQA, Zero-Shot)

## Scope
- Dataset: FinQA test split
- Decoding: greedy (`do_sample=False`)
- Default baseline: thinking=true + `[FINAL_ANSWER]...[/FINAL_ANSWER]` + `math_verify`
- Also reported: legacy numeric evaluator for side-by-side comparison

## Baseline Table (Latest per Model/Setting)
| model | setting | split | n | acc_mathverify | acc_legacy | delta | parse_fail_mv | parse_fail_legacy | tag_open_only_rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-4B | oracle | test | 1147 | 0.165650 | 0.000000 | 0.165650 | 0.006103 | 0.000000 | 0.000000 |
| Qwen/Qwen3-4B | full | test | 1147 | 0.136007 | 0.000000 | 0.136007 | 0.002616 | 0.000000 | 0.000000 |
| Qwen/Qwen3-8B | oracle | test | 1147 | 0.076722 | 0.000000 | 0.076722 | 0.007847 | 0.000000 | 0.000000 |
| Qwen/Qwen3-8B | full | test | 1147 | 0.044464 | 0.000000 | 0.044464 | 0.002616 | 0.000000 | 0.000000 |

## Key Findings
- Best math-verify run: Qwen/Qwen3-4B / oracle @ 0.1656
- Avg parse_fail (math-verify): 0.0048
- Avg parse_fail (legacy): 0.0000
- Tag-status distribution is recorded in `summary.json` (`tag_status_counts`) for truncation diagnostics.
- `open_only` tags indicate truncated close tags; extraction fallback still provides answer text.

## Artifacts
- `results/regression_final_answer_mathverify.md`
- `results/summary.json`
- `results/error_cases.md`