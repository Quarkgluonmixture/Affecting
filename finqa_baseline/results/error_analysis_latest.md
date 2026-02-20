# Error Analysis (Latest Runs)

- Total evaluated samples (4 runs combined): 4588
- Total wrong under adjusted metric: 3875

## Top Error Reasons
- percent_scale_miss: 1284 (33.1%)
- numeric_mismatch_other: 1084 (28.0%)
- spurious_small_integer: 577 (14.9%)
- magnitude_error_100x+: 466 (12.0%)
- sign_error: 239 (6.2%)
- gold_missing: 188 (4.9%)
- percent_scale_flip: 22 (0.6%)
- parse_fail: 14 (0.4%)

## Example Cases
### percent_scale_miss
- question: what percentage of total facilities as measured in square feet are leased?
- gold: 0.14
- pred: 100.0
- raw_output: (2.1 + 6.0) / 8.1 * 100

### numeric_mismatch_other
- question: what is the net change in net revenue during 2015 for entergy corporation?
- gold: 94.0
- pred: 84.0
- raw_output: 84

### spurious_small_integer
- question: what was the percentage change in total rental expense under operating leases from july 2 , 2005 to july 1 , 2006?
- gold: 0.07
- pred: 0.0
- raw_output: (100690000 - 92710000) / 92710000 *

### magnitude_error_100x+
- question: what would the fair value of total securities available for sale be without the fair value of securities classified as corporate stocks as of december 31 , 2012?
- gold: 61545.0
- pred: 18.0
- raw_output: 61018

### sign_error
- question: what is the net change in aon 2019s unpaid restructuring liabilities during 2006?
- gold: 18.0
- pred: -141.0
- raw_output: -141

