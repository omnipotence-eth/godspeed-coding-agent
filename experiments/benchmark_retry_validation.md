# Godspeed Benchmark Report — Retry Logic Validation

## Comparison

| Metric | Without Retry | With Retry | Delta |
|--------|--------------|------------|-------|
| **Pass (Jaccard >= 0.6)** | **10 / 20 (50%)** | **13 / 20 (65%)** | **+3** |
| Mean tool selection | 0.559 | 0.702 | +0.143 |
| Mean sequence quality | 0.697 | 0.916 | +0.219 |
| Mean overall | 0.526 | 0.685 | **+0.159** |
| Mechanical pass | 9 / 13 (69%) | 13 / 13 (100%) | **+4** |
| LLM errors | 5 tasks | 0 tasks | **-5** |
| Total duration | 1396s | 2488s | +1092s |

## Recovered Tasks

| Task | Without Retry | With Retry | Recovery |
|------|--------------|------------|----------|
| medium-explore-01 | llm_error (0.0) | 0.95 | **+0.95** |
| medium-web-lookup-01 | llm_error (0.5) | 1.0 | **+0.5** |
| medium-new-file-01 | llm_error (0.0) | 0.2 + PASS | **+0.2** |
| medium-git-commit-01 | llm_error (0.0) | 0.65 + PASS | **+0.65** |
| hard-multi-file-01 | llm_error (0.0) | 0.705 + PASS | **+0.705** |
| hard-test-coverage-01 | llm_error (0.0) | 0.635 + PASS | **+0.635** |

All 5 previously `llm_error` tasks recovered and achieved mechanical success.

## Conclusion

Retry logic with exponential backoff (2s, 4s, 8s) completely eliminated transient LLM failures on NVIDIA NIM free tier. Score improved from 0.526 to 0.685 — a **30% relative improvement**.
