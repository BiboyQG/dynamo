# Shared vLLM/TRT-LLM Scheduler

This directory implements the shared scheduler simulation for both vLLM and
TRT-LLM. `EngineType::Trtllm` uses the same `VllmCore` with a different
`SchedulingPolicy`.

## Ownership

- `core.rs` owns common queue mechanics, KV allocation, state transitions,
  completion, and event emission.
- `policy.rs` is the sole location for behavior that differs between vLLM and
  TRT-LLM.
- vLLM admits based on the current known sequence and permits decode-time
  preemption.
- TRT-LLM `GUARANTEED_NO_EVICT` reserves prompt plus maximum output and forbids
  preemption.

## Guardrails

- Do not recreate a separate TRT-LLM scheduler or scatter `EngineType` checks
  through `core.rs`; extend `SchedulingPolicy` and `policy.rs`.
- Compute prefix cost once per waiting candidate and pass it through the
  admission path. The waiting-policy view snapshots each candidate's cost after
  the running requests are scheduled; admission recomputes it because earlier
  admissions in the same pass change the active blocks.
- Discount only active cached prefix blocks from physical capacity. Inactive
  cached blocks must consume capacity when reactivated.
- The waiting queue is FIFO unless a `WaitingQueuePolicy` is installed; the
  policy only chooses the order in which a pass tries candidates. A candidate
  that cannot be admitted now must remain queued without preempting running
  work, and the policy's order is never written back to the queue.
