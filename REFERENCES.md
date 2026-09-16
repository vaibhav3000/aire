# References

## Concepts and prior art (ideas, not code)

- RAG (Retrieval-Augmented Generation) - Lewis et al., 2020, arXiv:2005.11401
- LLM-as-a-judge and its biases - Zheng et al., 2023, arXiv:2306.05685 (MT-Bench paper); position/verbosity/self-preference biases discussed in our docs
- HALT/truncation and abstention evaluation - Kamalloo et al., 2023, arXiv:2305.14695
- OpenAI Chat Completions API - https://platform.openai.com/docs/api-reference/chat (the OpenAI-compatible request/response shape our provider implements; Gemini exposes the same shape at generativelanguage.googleapis.com/v1beta/openai)
- DeepEval - https://github.com/confident-ai/deepeval (existing evaluation framework; we deliberately built the evaluator core from first principles instead - see README)
- Ragas - https://github.com/explodinggradients/ragas (RAG-focused evaluation framework; same reasoning)

## What is ours vs referenced

- All evaluation-engine code (schema, runner, metrics, judges, regression
  engine, reports) in src/aire/ is original, written for this project.
- The demo corpus (demo/corpus/*.md) is original fictional content.
- Ideas and metric definitions are informed by the public evaluation
  literature above; no text or code was copied from those sources.
