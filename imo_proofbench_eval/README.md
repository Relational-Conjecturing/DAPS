# IMO-ProofBench evaluation

`collect_novita.py` calls Novita AI for Qwen3-235B, DeepSeek-V3.1 and Kimi-K2-Instruct; `collect_gemini_flash.py` calls Gemini-3.6-Flash. The pipeline uses the original matched prompt, top-16 premises, eight answers per problem and API-default sampling. `grade_gemini.py` uses the original prompt with Gemini-2.5-Pro as the judge. Novita uses the compatible `openai` Python SDK.

Provide your own JSONL file with `Problem ID`, `Problem`, `Solution`, `Grading guidelines`, and a ranked `premise_list` containing names/signatures/statements. No input or result JSONL is bundled. Both collectors require `--input-jsonl FILE`; `--selector LABEL` optionally sets an output label. Collection skips duplicate problem/sample entries.

```bash
pip install -r ../requirements.txt
# Export NOVITA_API_KEY and GEMINI_API_KEY in your shell.
INPUT_JSONL=/absolute/path/to/input.jsonl OUTPUT_ROOT=/path/to/outputs RUN_DATE=run1 bash run_evaluation.sh all --dry-run
# Remove --dry-run to generate, grade and summarize; use qwen/deepseek/kimi/flash for one model.
```

`STAGE=collect|grade|summarize` reruns one step; keep the same `RUN_DATE`. Outputs include run settings, prompts, answers, judge scores and JSON/CSV summaries. No historical outputs are bundled.

To grade existing answers, pass their paths explicitly: `python grade_gemini.py --answers-jsonl /path/to/answers.jsonl --benchmark-jsonl /path/to/input.jsonl --dry-run`. For multiple selector inputs, run once per file with distinct `SELECTOR_LABEL` values; `summarize_scores.py` accepts a manifest listing graded file paths.

For the formalizer ablation, prepare your jsonl using different formalizer, and run `bash run_formalizer_ablation.sh A.jsonl B.jsonl C.jsonl`.

The reported statistic averages complete per-problem 0–7 judge scores, averages across problems and normalizes to a percentage; SE is across problem means.
