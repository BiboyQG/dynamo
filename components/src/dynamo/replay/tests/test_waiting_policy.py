# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waiting-queue policies on the offline aggregated replay.

The equality tests need bindings built with the replay-bench feature: only
then does the KV router seed its tie-breaking RNG and the loader assign
deterministic request ids. On an unseeded build they are skipped.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo._core import run_mocker_trace_replay
from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay, run_trace_replay
from dynamo.replay.waiting_policy import ScoreOrder, load

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.mocker,
]

POLICIES = Path(__file__).resolve().parents[1] / "policies"
TRACE_BLOCK_SIZE = 16
LOG_KEYS = {
    "now_ms",
    "worker_id",
    "num_running",
    "waiting_len",
    "free_blocks",
    "requests",
    "order",
}
REQUEST_KEYS = {
    "queue_index",
    "submit_seq",
    "wait_ms",
    "age_tokens",
    "prompt_tokens",
    "generated_tokens",
    "num_preemptions",
}
PREFIX_KEYS = REQUEST_KEYS | {"cached_prefix_tokens", "needed_blocks"}
CTX_KEYS = {
    "now_ms",
    "worker_id",
    "worker_role",
    "num_running",
    "max_num_seqs",
    "waiting_len",
    "block_size",
    "num_gpu_blocks",
    "free_blocks",
}
# Replay wall time and what is derived from it differ run to run.
WALL_TIME_KEYS = (
    "wall_time_ms",
    "processed_tokens_per_s",
    "processed_output_tokens_per_s",
    "waiting_policy_stats",
)


def engine_args(**overrides):
    """Four running requests per worker, so a burst queues and the policy is consulted."""
    args = {
        "block_size": TRACE_BLOCK_SIZE,
        "num_gpu_blocks": 256,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 2048,
    }
    args.update(overrides)
    return MockEngineArgs.from_json(json.dumps(args))


def write_trace(path, n=120, long_outputs=False):
    """A single-turn Mooncake burst: four prefix families, five prompt lengths.
    Long outputs (150 to 299 tokens) make a small KV pool preempt."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            family = i % 4
            input_length = 64 + 32 * (i % 5)
            blocks = input_length // TRACE_BLOCK_SIZE
            output_length = 150 + (i * 37) % 150 if long_outputs else 8 + i % 5
            f.write(
                json.dumps(
                    {
                        "timestamp": 3 * i,
                        "input_length": input_length,
                        "output_length": output_length,
                        "hash_ids": [family * 1000 + b for b in range(blocks)],
                    }
                )
                + "\n"
            )
    return path


@pytest.fixture
def trace(tmp_path):
    return write_trace(tmp_path / "trace.jsonl")


def replay(trace, tmp_path, name, **kwargs):
    records_path = tmp_path / f"{name}.jsonl"
    report = run_trace_replay(
        trace,
        extra_engine_args=kwargs.pop("extra_engine_args", engine_args()),
        trace_block_size=TRACE_BLOCK_SIZE,
        report_jsonl_path=records_path,
        **kwargs,
    )
    records = lines(records_path)
    return report, records


@pytest.fixture(scope="module")
def seeded_router(tmp_path_factory):
    """Two native 4-worker kv_router replays agree only on a replay-bench build."""
    tmp = tmp_path_factory.mktemp("seeded")
    probe = write_trace(tmp / "trace.jsonl")
    runs = [
        replay(probe, tmp, f"probe{i}", num_workers=4, router_mode="kv_router")[1]
        for i in range(2)
    ]
    if runs[0] != runs[1]:
        pytest.skip("needs bindings built with --features replay-bench")


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rounded(value):
    """Nanosecond precision: a few sums run in a different order on the
    single-worker and multi-worker runtimes."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def comparable(report, records, single_worker=False):
    summary = {k: v for k, v in report.items() if k not in WALL_TIME_KEYS}
    records = sorted(records, key=lambda r: (r["arrival_time_ms"], r["input_length"]))
    if single_worker:
        # The native one-worker replay runs on SingleRuntime, which leaves the
        # worker indices unset; a policy moves it to the multi-worker runtime.
        for record in records:
            del record["decode_worker_idx"], record["prefill_worker_idx"]
    return rounded(summary), rounded(records)


@pytest.mark.parametrize(
    ("num_workers", "router_mode"), [(1, "round_robin"), (4, "kv_router")]
)
def test_fcfs_matches_native_order(
    seeded_router, trace, tmp_path, num_workers, router_mode
):
    native = replay(
        trace, tmp_path, "native", num_workers=num_workers, router_mode=router_mode
    )
    log = tmp_path / "decisions.jsonl"
    fcfs = replay(
        trace,
        tmp_path,
        "fcfs",
        num_workers=num_workers,
        router_mode=router_mode,
        waiting_policy=POLICIES / "fcfs.py",
        waiting_policy_log=log,
    )
    assert native[0]["completed_requests"] == 120
    single = num_workers == 1
    assert comparable(*native, single_worker=single) == comparable(
        *fcfs, single_worker=single
    )
    assert fcfs[0]["waiting_policy_stats"]["calls"] > 0
    rows = lines(log)
    assert fcfs[0]["waiting_policy_stats"]["max_waiting_len"] == max(
        line["waiting_len"] for line in rows
    )
    assert {line["worker_id"] for line in rows} == set(range(num_workers))
    # Each line is the view the policy saw and the order it chose.
    for line in rows:
        assert set(line) == LOG_KEYS
        assert len(line["requests"]) == line["waiting_len"] >= 2
        assert all(set(r) == REQUEST_KEYS for r in line["requests"])
        assert sorted(line["order"]) == sorted(
            r["submit_seq"] for r in line["requests"]
        )
        # fcfs keeps the native order.
        assert line["order"] == [r["submit_seq"] for r in line["requests"]]


def test_spf_reorders_and_is_deterministic(seeded_router, trace, tmp_path):
    native = replay(trace, tmp_path, "native", num_workers=4, router_mode="kv_router")
    spf = load(POLICIES / "spf.py")
    first = replay(
        trace,
        tmp_path,
        "spf1",
        num_workers=4,
        router_mode="kv_router",
        waiting_policy=spf,
    )
    second = replay(
        trace,
        tmp_path,
        "spf2",
        num_workers=4,
        router_mode="kv_router",
        waiting_policy=spf,
    )
    assert first[0]["completed_requests"] == native[0]["completed_requests"] == 120
    assert first[0]["mean_ttft_ms"] != native[0]["mean_ttft_ms"]
    assert comparable(*first) == comparable(*second)


@pytest.mark.parametrize("name", ["sjf", "lpm", "hrrn"])
def test_baseline_policies_complete_the_trace(trace, tmp_path, name):
    report, _ = replay(
        trace, tmp_path, name, num_workers=2, waiting_policy=POLICIES / f"{name}.py"
    )
    assert report["completed_requests"] == 120
    assert report["waiting_policy_stats"]["calls"] > 0


class PreemptionCheck(ScoreOrder):
    """Asserts that every view with preempted requests puts them first, in
    their native (queue_index) order."""

    views_with_preempted = 0

    def order(self, view):
        requests = list(view["requests"])
        order = super().order(view)
        preempted = [i for i, r in enumerate(requests) if r["num_preemptions"] > 0]
        if preempted:
            self.views_with_preempted += 1
            native = sorted(preempted, key=lambda i: requests[i]["queue_index"])
            assert order[: len(preempted)] == native, (order, requests)
        return order


@pytest.mark.parametrize("name", ["sjf", "spf", "lpm"])
def test_preempted_requests_go_back_first(tmp_path, name):
    trace = write_trace(tmp_path / "long.jsonl", n=80, long_outputs=True)
    policy = PreemptionCheck(load(POLICIES / f"{name}.py"))
    report = run_mocker_trace_replay(
        [str(trace)],
        extra_engine_args=engine_args(num_gpu_blocks=64, max_num_seqs=8),
        num_workers=1,
        trace_block_size=TRACE_BLOCK_SIZE,
        waiting_policy=policy,
    )
    assert report["completed_requests"] == 80
    assert policy.views_with_preempted > 0


def test_score_order_contract():
    def order_of(scores, submit_seqs):
        module = SimpleNamespace(uses_prefix_hits=False, score=lambda r, ctx: r["s"])
        requests = [{"submit_seq": q, "s": s} for q, s in zip(submit_seqs, scores)]
        return ScoreOrder(module).order({"requests": requests})

    # NaN sorts last; equal scores by submit_seq.
    assert order_of([float("nan"), 1.0, 1.0], [5, 9, 3]) == [2, 1, 0]
    # Integers are compared exactly, not through float.
    assert order_of([10**18 + 7, 10**18 + 5], [1, 2]) == [1, 0]
    with pytest.raises(TypeError, match="score must return a number"):
        order_of([True, 0.0], [1, 2])
    # The log write is outside the timed part.
    slow_log = SimpleNamespace(write=lambda text: time.sleep(0.05))
    module = SimpleNamespace(uses_prefix_hits=False, score=lambda r, ctx: 0.0)
    scorer = ScoreOrder(module, decision_log=slow_log)
    ctx = {
        "now_ms": 0.0,
        "worker_id": 0,
        "num_running": 0,
        "waiting_len": 1,
        "free_blocks": 0,
    }
    scorer.order({**ctx, "requests": [{"submit_seq": 1}]})
    assert scorer.stats["seconds"] < 0.05


def test_load_registers_the_module_so_dataclasses_resolve_annotations(tmp_path):
    policy = tmp_path / "typed.py"
    policy.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Weights:\n"
        "    prompt: float\n"
        "W = Weights(prompt=2.0)\n"
        "def score(request, ctx):\n"
        "    return W.prompt * request['prompt_tokens']\n"
    )
    assert load(policy).score({"prompt_tokens": 3}, {}) == 6.0


SUBPROCESS = """
import json, sys
from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_trace_replay
trace, policy, log, out, args, block = sys.argv[1:7]
report = run_trace_replay(
    trace, extra_engine_args=MockEngineArgs.from_json(args), num_workers=4,
    router_mode="kv_router", trace_block_size=int(block),
    waiting_policy=policy, waiting_policy_log=log,
)
skip = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s",
        "waiting_policy_stats"}
with open(out, "w") as f:
    json.dump({k: v for k, v in report.items() if k not in skip}, f, sort_keys=True)
"""


def test_report_and_log_do_not_depend_on_the_hash_seed(seeded_router, trace, tmp_path):
    outputs = []
    for seed in ("1", "2"):
        out = tmp_path / f"report-{seed}.json"
        log = tmp_path / f"log-{seed}.jsonl"
        env = dict(os.environ, PYTHONHASHSEED=seed)
        subprocess.run(
            [
                sys.executable,
                "-c",
                SUBPROCESS,
                str(trace),
                str(POLICIES / "spf.py"),
                str(log),
                str(out),
                json.dumps(
                    {
                        "block_size": TRACE_BLOCK_SIZE,
                        "num_gpu_blocks": 256,
                        "max_num_seqs": 4,
                        "max_num_batched_tokens": 2048,
                    }
                ),
                str(TRACE_BLOCK_SIZE),
            ],
            env=env,
            check=True,
        )
        outputs.append((out.read_text(), log.read_text()))
    assert outputs[0] == outputs[1]
    first_line = json.loads(outputs[0][1].splitlines()[0])
    assert set(first_line) == LOG_KEYS


def test_policy_exception_propagates(trace, tmp_path):
    policy = tmp_path / "broken.py"
    policy.write_text("def score(request, ctx):\n    return 1 / 0\n")
    with pytest.raises(ZeroDivisionError):
        run_trace_replay(
            trace,
            extra_engine_args=engine_args(),
            trace_block_size=TRACE_BLOCK_SIZE,
            waiting_policy=policy,
        )


def test_non_numeric_score_is_a_type_error(trace, tmp_path):
    policy = tmp_path / "strings.py"
    policy.write_text("def score(request, ctx):\n    return 'first'\n")
    with pytest.raises(TypeError, match="score must return a number"):
        run_trace_replay(
            trace,
            extra_engine_args=engine_args(),
            trace_block_size=TRACE_BLOCK_SIZE,
            waiting_policy=policy,
        )


class NoScaling:
    """A scaling policy whose first tick never comes."""

    def initial_tick_ms(self):
        return float("inf")

    def on_tick(self, snapshot):
        raise AssertionError("never ticks")


def test_scaling_and_waiting_policies_together(trace, tmp_path):
    report = run_mocker_trace_replay(
        [str(trace)],
        extra_engine_args=engine_args(),
        num_workers=2,
        trace_block_size=TRACE_BLOCK_SIZE,
        scaling_policy=NoScaling(),
        waiting_policy=ScoreOrder(load(POLICIES / "spf.py")),
    )
    assert report["completed_requests"] == 120
    broken = tmp_path / "broken.py"
    broken.write_text("def score(request, ctx):\n    return 1 / 0\n")
    with pytest.raises(ZeroDivisionError):
        run_mocker_trace_replay(
            [str(trace)],
            extra_engine_args=engine_args(),
            num_workers=2,
            trace_block_size=TRACE_BLOCK_SIZE,
            scaling_policy=NoScaling(),
            waiting_policy=ScoreOrder(load(broken)),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"replay_mode": "online"}, "replay_mode='offline'"),
        ({"replay_concurrency": 8}, "replay_concurrency"),
        ({"trace_format": "dynamo"}, "trace_format='mooncake'"),
        (
            {
                "extra_engine_args": None,
                "prefill_engine_args": engine_args(worker_type="prefill"),
                "decode_engine_args": engine_args(worker_type="decode"),
            },
            "aggregated",
        ),
    ],
)
def test_unsupported_combinations_are_rejected(trace, kwargs, message):
    kwargs.setdefault("extra_engine_args", engine_args())
    with pytest.raises(ValueError, match=message):
        run_trace_replay(
            trace,
            trace_block_size=TRACE_BLOCK_SIZE,
            waiting_policy=POLICIES / "fcfs.py",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"replay_mode": "online"}, "replay_mode='offline'"),
        ({"replay_concurrency": 4}, "replay_concurrency"),
        (
            {
                "extra_engine_args": None,
                "prefill_engine_args": engine_args(worker_type="prefill"),
                "decode_engine_args": engine_args(worker_type="decode"),
            },
            "aggregated",
        ),
    ],
)
def test_synthetic_unsupported_combinations_are_rejected(kwargs, message):
    kwargs.setdefault("extra_engine_args", engine_args())
    with pytest.raises(ValueError, match=message):
        run_synthetic_trace_replay(
            128,
            8,
            20,
            arrival_interval_ms=3.0,
            waiting_policy=POLICIES / "fcfs.py",
            **kwargs,
        )


def test_log_carries_prefix_fields_and_the_policy_order(trace, tmp_path):
    # The log changes nothing: same report as the same policy without a log.
    plain, _ = replay(trace, tmp_path, "lpm-plain", waiting_policy=POLICIES / "lpm.py")
    log = tmp_path / "decisions.jsonl"
    logged, _ = replay(
        trace,
        tmp_path,
        "lpm",
        waiting_policy=POLICIES / "lpm.py",
        waiting_policy_log=log,
    )
    for key in WALL_TIME_KEYS:
        plain.pop(key, None)
        logged.pop(key, None)
    assert rounded(plain) == rounded(logged)
    rows = lines(log)
    assert rows
    assert all(set(line) == LOG_KEYS for line in rows)
    assert all(set(r) == PREFIX_KEYS for line in rows for r in line["requests"])
    # lpm's order among new requests (this trace never preempts): most cached
    # prefix first, ties by submit_seq.
    for line in rows:
        by_seq = {r["submit_seq"]: r for r in line["requests"]}
        assert sorted(line["order"]) == sorted(by_seq)
        key = [(-by_seq[seq]["cached_prefix_tokens"], seq) for seq in line["order"]]
        assert key == sorted(key)


@pytest.mark.parametrize("name", ["sjf", "spf", "hrrn"])
def test_baselines_order_by_their_documented_keys(trace, tmp_path, name):
    # The key each docstring states, recomputed from the logged view; this trace
    # never preempts, so sjf's and spf's preempted-first clause does not come into play.
    def key(r):
        known = r["prompt_tokens"] + r["generated_tokens"]
        if name == "sjf":
            return float(r["prompt_tokens"])
        if name == "spf":
            return float(known - r["cached_prefix_tokens"])
        service = max(known - r["cached_prefix_tokens"], 1)
        return -(r["age_tokens"] + service) / service

    log = tmp_path / "decisions.jsonl"
    replay(
        trace,
        tmp_path,
        name,
        waiting_policy=POLICIES / f"{name}.py",
        waiting_policy_log=log,
    )
    rows = lines(log)
    assert rows
    distinct = 0
    for line in rows:
        assert all(r["num_preemptions"] == 0 for r in line["requests"])
        by_seq = {r["submit_seq"]: r for r in line["requests"]}
        assert sorted(line["order"]) == sorted(by_seq)
        keys = [(key(by_seq[seq]), seq) for seq in line["order"]]
        assert keys == sorted(keys)
        distinct = max(distinct, len({k for k, _ in keys}))
    assert distinct > 1


def test_log_shows_the_requests_before_the_policy_touched_them(trace, tmp_path):
    policy = tmp_path / "mutating.py"
    policy.write_text(
        "def score(request, ctx):\n"
        f"    assert set(ctx) == {CTX_KEYS!r}\n"
        "    request['prompt_tokens'] = -1\n"
        "    ctx['now_ms'] = -1.0\n"
        "    return float(request['queue_index'])\n"
    )
    log = tmp_path / "decisions.jsonl"
    # A reused log path is truncated.
    log.write_text("stale\n")
    replay(trace, tmp_path, "mutating", waiting_policy=policy, waiting_policy_log=log)
    rows = lines(log)
    assert rows
    assert all(r["prompt_tokens"] > 0 for line in rows for r in line["requests"])
    assert all(line["now_ms"] >= 0 for line in rows)
    # The line shape, on any build: one worker, a base policy.
    for line in rows:
        assert set(line) == LOG_KEYS
        assert all(set(r) == REQUEST_KEYS for r in line["requests"])
        assert [r["queue_index"] for r in line["requests"]] == list(
            range(line["waiting_len"])
        )
        assert sorted(line["order"]) == sorted(
            r["submit_seq"] for r in line["requests"]
        )


def test_log_without_policy_is_rejected(trace, tmp_path):
    with pytest.raises(ValueError, match="waiting_policy_log needs waiting_policy"):
        run_trace_replay(
            trace,
            extra_engine_args=engine_args(),
            trace_block_size=TRACE_BLOCK_SIZE,
            waiting_policy_log=tmp_path / "log.jsonl",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"engine_type": "trtllm"}, "vLLM scheduling policy"),
        ({"max_num_seqs": None}, "explicit max_num_seqs"),
    ],
)
def test_engine_rejections_come_from_the_scheduler(trace, overrides, message):
    with pytest.raises(Exception, match=message):
        run_trace_replay(
            trace,
            extra_engine_args=engine_args(**overrides),
            trace_block_size=TRACE_BLOCK_SIZE,
            waiting_policy=POLICIES / "fcfs.py",
        )


@pytest.mark.parametrize("turns_per_session", [1, 2])
def test_synthetic_replay_takes_a_policy(turns_per_session):
    report = run_synthetic_trace_replay(
        128,
        8,
        60,
        extra_engine_args=engine_args(),
        num_workers=2,
        arrival_interval_ms=3.0,
        turns_per_session=turns_per_session,
        waiting_policy=POLICIES / "spf.py",
    )
    assert report["completed_requests"] == 60 * turns_per_session
    assert report["waiting_policy_stats"]["calls"] > 0
