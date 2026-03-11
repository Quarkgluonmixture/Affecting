# Regression Check: FINAL_ANSWER + Math-Verify

| case | passed | tag_status | mathverify_correct | mathverify_parse_fail | legacy_pred | legacy_correct |
|---|---|---|---|---|---:|---|
| multi-number noise, tagged answer wins | True | closed | True | False | 37.1 | True |
| truncated output with open tag only | True | open_only | True | False | 0.42 | True |
| percent vs decimal equivalence | True | closed | True | False | 0.42 | True |
| no tag fallback does not crash | True | absent | True | False | 3.5 | True |

- pass_count: 4/4