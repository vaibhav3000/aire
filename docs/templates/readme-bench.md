Measured against `FakeLLMServer`, a rate-limited fake LLM endpoint on 127.0.0.1, not a real provider: {{bench.server.requests_per_minute|.0f}} requests per minute, {{bench.server.service_time_s}} s per request and a retry hint on every 429. One run of each configuration over the {{bench.n_cases}}-case suite, recorded at {{bench.recorded_at_utc}} in [results/runner_benchmark.json](results/runner_benchmark.json):

| Configuration | Wall time (s) | Requests sent | 429 responses | Failed cases |
| --- | --- | --- | --- | --- |
| 1 worker | {{bench.sequential.wall_time_s|.3f}} | {{bench.sequential.server.requests}} | {{bench.sequential.server.rate_limited}} | {{bench.sequential.n_failed}} |
| 8 workers | {{bench.concurrent.wall_time_s|.3f}} | {{bench.concurrent.server.requests}} | {{bench.concurrent.server.rate_limited}} | {{bench.concurrent.n_failed}} |
| 8 workers, client told twice the server's limit | {{bench.over_eager.wall_time_s|.3f}} | {{bench.over_eager.server.requests}} | {{bench.over_eager.server.rate_limited}} | {{bench.over_eager.n_failed}} |
| 8 workers, warm response cache | {{bench.warm_cache.wall_time_s|.3f}} | {{bench.warm_cache.server.requests}} | {{bench.warm_cache.server.rate_limited}} | {{bench.warm_cache.n_failed}} |

Eight workers finish {{bench.concurrent.speedup|.1f}} times faster than one. With its budget set too high, the client met {{bench.over_eager.server.rate_limited}} rate-limit responses, backed off as the server asked, and still answered every case. The warm cache answers the suite without a single request. The timings are from one machine ({{bench.machine.platform}}, Python {{bench.machine.python}}); `make check` does not compare them, and `python demo/bench_runner.py` records a fresh run.
