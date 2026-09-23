# GLM-5.3 Flash output duplication investigation

Date: 2026-09-22. Host: our rig. Endpoint: localhost:8002.

## Finding

The reproduced `17 + 25 = 4242` response is caused by a mismatch between the
installed chat template and GLM reasoning parser when a request supplies
`chat_template_kwargs: {"enable_thinking": false}`.

The model template at `/mnt/ssd/models/GLM-5.3-Flash-AWQ-W4A16/chat_template.jinja`
does not use `enable_thinking` and always appends `<think>` to the assistant
generation prompt. In contrast, `Glm47MoeParser` in
`vllm/parser/glm47_moe.py` uses that flag to disable reasoning parsing.
Consequently, reasoning text and final answer are returned together as content.

The raw generated sequence for the arithmetic prompt is
`17 + 25 = 42</think>42<|user|>`: the answer appears once in reasoning and once in
the final response. A raw completions request using the same prompt token IDs
confirmed that boundary. This evidence explains the observed duplication
without attributing it to speculative decoding. DFlash was enabled throughout;
a speculative-versus-ordinary generation comparison was not performed.

## Verified request workaround

Remove `chat_template_kwargs.enable_thinking=false`. To return only the final
answer, use the top-level request field `"include_reasoning": false` instead:

```json
{
  "model": "glm53-flash",
  "messages": [{"role": "user", "content": "What is 17 plus 25? Give only the number."}],
  "temperature": 0,
  "max_tokens": 256,
  "include_reasoning": false
}
```

This hides reasoning in the response; it does not disable its generation or
remove its token cost. Do not retain the unsupported false flag alongside it.

## Validation

`verify_output_modes.py` executed 18 requests: three prompts (42, 4, Paris),
three request modes, and streaming/non-streaming responses.

* Unsupported disable flag: all six responses mixed reasoning and answer.
* Default request: all six returned exactly the expected final answer in content,
  with reasoning in its separate field.
* `include_reasoning=false`, without the disable flag: all six returned exactly
  the expected answer and no reasoning field content.

The earlier `duplication_probe.py` captured five prompts in both thinking flag
modes, including generated token IDs, plus a raw completion of the arithmetic
prompt. Evidence files are `dflash.jsonl` and `response-modes.jsonl` in this
report directory. The verification script asserts expected content and absent
reasoning for the workaround; it completed successfully.

## Scope and remaining work

No serving code, model template, or client configuration was changed during
this phase. No server restart or driver reload was performed. Fable is working
on TurboQuant in parallel; coordinate server lifecycle changes with the user.

This is a verified request-level workaround, not a server-side patch. The
server still mishandles the unsupported false flag. A general fix should keep
parser state consistent with the rendered prompt and cover genuinely switchable
templates as well as always-reasoning templates. An actual client emitting the
flag has not been identified or modified. Other forms of repeated generated
text, if present, require separate reproduction.

The previous deep-prefill fix is now committed as `69dc4f43c8`; this report does
not change it.
