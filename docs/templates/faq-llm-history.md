re-scored. It answered "I don't know." on all {{llm_hist.n_unanswerable|word}} unanswerable questions,
where both deterministic versions answer, which is a real but {{llm_hist.n_unanswerable|word}}-case
observation. Its groundedness of {{llm_hist.mean_groundedness|.1f}} averaged only the {{llm_hist.n_grounded}} cases that made a
claim, against a deterministic baseline that the citation-placement bug had
pushed down to {{cmp.llm_hist.groundedness.baseline|.4f}}, so there was no measured grounding gain. Its {{llm_hist.mean_latency_s|.1f}} s
