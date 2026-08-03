# Qwen3 thinking mode is disabled everywhere

Qwen3-8B defaults to emitting a `<think>` block before its answer. Under GRPO
with `max_completion_length=512` this reliably consumed the entire budget before
the model ever produced `MOVE:`, so every generation scored as a format failure
and training could not start. Thinking mode is therefore switched off at every
layer: `enable_thinking=False` in `apply_chat_template()` for training and Python
inference, and `/no_think` in the system prompt plus
`"chat_template_kwargs": {"enable_thinking": false}` plus a generous `max_tokens`
in the LM Studio API calls made by the GTP proxy.

This is worth recording because the model still *can* think and someone will
reasonably wonder why we suppress it on a reasoning task. The v4 prompt asks for
explicit `REASONING:` text instead — same benefit, inside the token budget, and
parseable.

## Consequences

Three separate places must be kept in sync. The LM Studio path is the fragile
one: even with `enable_thinking=False`, some versions route output into
`reasoning_content` instead of `content`, which surfaces as empty responses
rather than as an error.
