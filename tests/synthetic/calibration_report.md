# Synthetic Calibration Report

Date: 2026-05-30T19:15:50.390215+00:00
Workflows tested: 520
Daily volume: 1000 requests/day

## Overall Calibration

- p50 within (0.7, 2.0): 91%
- p95 coverage >= true: 78%
- Mean |p50 error|: 0.162

## By Sample Size

| n | p50 calib | p95 coverage | Mean |p50 err| |
|---|-----------|-------------|-----------------|
| 20 | 72% | 88% | 0.267 |
| 50 | 89% | 67% | 0.192 |
| 100 | 96% | 76% | 0.143 |
| 150 | 96% | 74% | 0.121 |
| 300 | 100% | 82% | 0.088 |

## By Distribution Type

| Type | p50 calib | p95 coverage |
|------|-----------|-------------|
| bimodal | 92% | 77% |
| lognormal | 96% | 80% |
| pareto | 80% | 82% |
| uniform | 100% | 100% |
| zero_inflated | 80% | 71% |

## Failures (48 workflows)

| Workflow | p50 ratio | Type | n |
|----------|-----------|------|---|
| lognormal_sigma_1.0_n_20 | 0.57 | lognormal | 20 |
| lognormal_sigma_1.3_n_20 | 0.374 | lognormal | 20 |
| lognormal_sigma_1.5_n_20 | 0.194 | lognormal | 20 |
| bimodal_mix_0.05_sep_2_sw_0.5_n_20 | 0.66 | bimodal | 20 |
| bimodal_mix_0.05_sep_5_sw_0.5_n_50 | 0.692 | bimodal | 50 |
| bimodal_mix_0.05_sep_10_sw_0.2_n_20 | 0.658 | bimodal | 20 |
| bimodal_mix_0.05_sep_10_sw_0.2_n_50 | 0.68 | bimodal | 50 |
| bimodal_mix_0.05_sep_10_sw_0.2_n_150 | 0.667 | bimodal | 150 |
| bimodal_mix_0.05_sep_10_sw_0.5_n_20 | 0.655 | bimodal | 20 |
| bimodal_mix_0.05_sep_15_sw_0.2_n_20 | 0.546 | bimodal | 20 |
| bimodal_mix_0.05_sep_15_sw_0.2_n_50 | 0.562 | bimodal | 50 |
| bimodal_mix_0.05_sep_15_sw_0.5_n_20 | 0.498 | bimodal | 20 |
| bimodal_mix_0.05_sep_15_sw_0.5_n_50 | 0.595 | bimodal | 50 |
| bimodal_mix_0.05_sep_15_sw_0.5_n_100 | 0.539 | bimodal | 100 |
| bimodal_mix_0.05_sep_20_sw_0.2_n_20 | 0.516 | bimodal | 20 |
| bimodal_mix_0.05_sep_20_sw_0.5_n_20 | 0.381 | bimodal | 20 |
| bimodal_mix_0.1_sep_3_sw_0.5_n_20 | 0.595 | bimodal | 20 |
| bimodal_mix_0.1_sep_5_sw_0.2_n_20 | 0.669 | bimodal | 20 |
| bimodal_mix_0.1_sep_5_sw_0.5_n_20 | 0.62 | bimodal | 20 |
| bimodal_mix_0.1_sep_10_sw_0.2_n_20 | 0.524 | bimodal | 20 |
| ... and 28 more | | | |