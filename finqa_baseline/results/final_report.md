# Final Baseline Report (FinQA, Zero-Shot)

## Scope
- Dataset: FinQA test split
- Metrics: base accuracy, adjusted accuracy (percent auto-scale), parse fail rate
- Decoding: greedy (`do_sample=False`)
- Prompt policy: numeric-only answer; thinking disabled by default

## Baseline Table (Latest per Model/Setting)
| model | setting | split | n | acc_base | acc_adjusted | delta | parse_fail_rate | recovered |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-8B | oracle | test | 1147 | 0.076722 | 0.108108 | 0.031386 | 0.007847 | 36 |
| Qwen/Qwen3-8B | full | test | 1147 | 0.044464 | 0.059285 | 0.014821 | 0.002616 | 17 |
| Qwen/Qwen3-4B | oracle | test | 1147 | 0.165650 | 0.258065 | 0.092415 | 0.006103 | 106 |
| Qwen/Qwen3-4B | full | test | 1147 | 0.136007 | 0.196164 | 0.060157 | 0.002616 | 69 |

## Key Findings
- Best adjusted run: Qwen/Qwen3-4B / oracle @ 0.2581
- Worst adjusted run: Qwen/Qwen3-8B / full @ 0.0593
- Percent auto-scale recovers a non-trivial number of errors, especially for 4B.
- Parse failures are now low; dominant failures are numeric reasoning and scale/selection mistakes.

## Error Breakdown (Aggregated Latest Runs)
- percent_scale_miss: 1284 (33.1%)
- numeric_mismatch_other: 1084 (28.0%)
- spurious_small_integer: 577 (14.9%)
- magnitude_error_100x+: 466 (12.0%)
- sign_error: 239 (6.2%)
- gold_missing: 188 (4.9%)
- percent_scale_flip: 22 (0.6%)
- parse_fail: 14 (0.4%)
- year_number_leak: 1 (0.0%)

## Interpretation
- Disabling thinking mode removed most parse-fail problems from long thought traces.
- Remaining gap is primarily answer correctness, not answer extractability.
- Full-context setting can still hurt due to distractor context and wrong number selection.

## Next-Step Recommendations
1. Keep adjusted and base metrics both reported for transparency.
2. Add stricter output format constraints (single numeric token regex guard).
3. Introduce lightweight verifier/re-ranking after baseline reporting is finalized.
4. Evaluate per-question-type slices (percent, ratio, difference, multi-step arithmetic).
