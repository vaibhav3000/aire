# The aire trace stays canonical; OpenTelemetry is an import and export format

Status: accepted (Step 2)

AIRE needs a record that is complete, versioned and replayable: the exact
answer, retrieved documents, tool calls and token source for every case. An
OpenTelemetry pipeline samples, batches and may drop spans, and the GenAI
semantic conventions are still in Development status. So `traces.jsonl` remains
the source of truth, and OpenTelemetry spans are produced from completed traces
(with their recorded timestamps) and can be ingested back into aire traces from
OTLP JSON. The mapping, including every `gen_ai.*` attribute name, lives in one
module (`aire.telemetry.genai`) so a convention change is a one-file edit.
The trade-off: spans are emitted after a case finishes rather than live, so they
do not nest under a host application's in-flight request span. For an
evaluation harness that is acceptable; live instrumentation remains possible by
calling the same mapping from inside an application.
