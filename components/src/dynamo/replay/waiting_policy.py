# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waiting-queue policies for offline aggregated replay.

A policy is a Python module with::

    uses_prefix_hits = False   # optional; True adds the two prefix fields

    def score(request: dict, ctx: dict) -> float: ...

Each scheduler pass that would admit waiting requests calls ``score`` once per
live waiting request, after the running requests are scheduled and before the
admission loop, and tries the requests in ascending score, ties by
``submit_seq``; NaN counts as +inf. The admission rules do not change: the first
request that does not fit stops the loop, nothing is preempted for a candidate,
and the order is not written back to the queue (a fresh request whose known
sequence exceeds the whole KV pool, possible only on a deployment vLLM refuses,
is rejected and skipped instead). The pass keeps the native FCFS
order, without calling the policy, when the loop would not run (a preemption
this pass, running set full, no token budget), when fewer than two requests
wait, and, for a prefix-aware policy, when more than 128 wait.

``request`` fields:

- ``queue_index``: position among the live waiting requests in native order
  (preempted requests first).
- ``submit_seq``: arrival order at this scheduler, the tie key; a preemption
  keeps it.
- ``wait_ms``: virtual time since the request reached this worker's queue; a
  preemption does not reset it.
- ``age_tokens``: tokens of the passes this scheduler started since the
  request reached its queue; the pass running at arrival counts toward
  ``wait_ms`` but not here. A preemption does not reset it.
- ``prompt_tokens``: prompt length.
- ``generated_tokens``: tokens generated before a preemption; 0 for a new
  request.
- ``num_preemptions``: times the request was preempted.
- ``cached_prefix_tokens`` (``uses_prefix_hits`` only): prefix-cache hit on
  the known sequence, whole blocks, at most its length minus one.
- ``needed_blocks`` (``uses_prefix_hits`` only): KV blocks the admission gate
  requires to be free, the known sequence minus the active cached prefix;
  ``needed_blocks <= ctx["free_blocks"]`` is the admission test.

``ctx`` fields: ``now_ms``, ``worker_id`` (opaque, one per scheduler),
``worker_role`` (``"aggregated"``), ``num_running``, ``max_num_seqs``,
``waiting_len``, ``block_size``, ``num_gpu_blocks``, ``free_blocks`` (blocks not
held by running requests; evictable cached blocks count as free).

Sequence lengths and prefix hits are in tokens, KV capacity in blocks. The
fields are a snapshot taken after the running requests are scheduled, before
the admission loop; the simulated clock does not advance while the policy runs.
"""

import importlib.util
import json
import math
import numbers
import os
import sys
import time
from types import ModuleType


def load(path) -> ModuleType:
    """Import a policy file as a module."""
    path = os.fspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"waiting_policy_{name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} is not a Python source file")
    module = importlib.util.module_from_spec(spec)
    # Registered so that dataclasses and `from __future__ import annotations`
    # work inside the policy file.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScoreOrder:
    """What the replay binding calls once per pass: scores every waiting
    request with the module's ``score`` and returns the admission order as
    indices into ``view["requests"]``.

    With a ``decision_log`` (a writable text file) every call appends one JSON
    line: the per-pass ``ctx`` fields (``now_ms``, ``worker_id``,
    ``num_running``, ``waiting_len``, ``free_blocks``), then ``requests``, the
    waiting requests exactly as the policy received them, in native order with
    every field, and ``order``, the ``submit_seq`` of each request in the order
    the policy chose. The other ``ctx`` fields (``worker_role``,
    ``max_num_seqs``, ``block_size``, ``num_gpu_blocks``) are the engine
    arguments and are not written. ``stats["seconds"]`` covers scoring and the
    sort, not the log."""

    def __init__(self, module: ModuleType, decision_log=None):
        self.score = module.score
        self.uses_prefix_hits = bool(getattr(module, "uses_prefix_hits", False))
        self.decision_log = decision_log
        self.stats = {"calls": 0, "seconds": 0.0, "max_waiting_len": 0}

    def order(self, view: dict) -> list[int]:
        requests = view.pop("requests")
        ctx = view
        # Copied before scoring: a policy may mutate the dicts it is given.
        log_line = None
        if self.decision_log is not None:
            log_line = {
                "now_ms": ctx["now_ms"],
                "worker_id": ctx["worker_id"],
                "num_running": ctx["num_running"],
                "waiting_len": ctx["waiting_len"],
                "free_blocks": ctx["free_blocks"],
                "requests": [dict(request) for request in requests],
            }
        started = time.perf_counter()
        keyed = []
        for index, request in enumerate(requests):
            submit_seq = request["submit_seq"]
            score = self.score(request, ctx)
            if isinstance(score, bool) or not isinstance(score, numbers.Real):
                raise TypeError(
                    f"score must return a number, got {type(score).__name__}"
                )
            if score != score:  # NaN; kept as returned otherwise so ints stay exact
                score = math.inf
            keyed.append((score, submit_seq, index))
        keyed.sort()
        self.stats["calls"] += 1
        self.stats["seconds"] += time.perf_counter() - started
        self.stats["max_waiting_len"] = max(
            self.stats["max_waiting_len"], len(requests)
        )
        if log_line is not None:
            log_line["order"] = [submit_seq for _, submit_seq, _ in keyed]
            self.decision_log.write(json.dumps(log_line) + "\n")
        return [index for _, _, index in keyed]
