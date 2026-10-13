# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-iteration FPM log on the offline aggregated replay."""

import json
from itertools import pairwise

import pytest

from dynamo.replay import run_synthetic_trace_replay, run_trace_replay
from dynamo.replay.tests.test_waiting_policy import (
    TRACE_BLOCK_SIZE,
    WALL_TIME_KEYS,
    engine_args,
    rounded,
    write_trace,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.mocker,
]

FIELDS = {
    "now_ms",
    "worker_id",
    "dp_rank",
    "wall_time",
    "num_prefill_requests",
    "sum_prefill_tokens",
    "var_prefill_length",
    "sum_prefill_kv_tokens",
    "num_decode_requests",
    "sum_decode_kv_tokens",
    "var_decode_kv_tokens",
    "num_queued_prefill",
    "sum_queued_prefill_tokens",
    "var_queued_prefill_length",
    "num_queued_decode",
    "sum_queued_decode_kv_tokens",
    "var_queued_decode_kv_tokens",
}


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_line_per_iteration_and_the_report_is_unchanged(tmp_path):
    # No prefix caching, so every prompt token is prefilled exactly once.
    trace = write_trace(tmp_path / "trace.jsonl")
    plain = run_trace_replay(
        trace,
        extra_engine_args=engine_args(enable_prefix_caching=False),
        trace_block_size=TRACE_BLOCK_SIZE,
        num_workers=2,
    )
    log = tmp_path / "workers.jsonl"
    logged = run_trace_replay(
        trace,
        extra_engine_args=engine_args(enable_prefix_caching=False),
        trace_block_size=TRACE_BLOCK_SIZE,
        num_workers=2,
        fpm_log=log,
    )
    for key in WALL_TIME_KEYS:
        plain.pop(key, None)
        logged.pop(key, None)
    assert rounded(plain) == rounded(logged)

    rows = lines(log)
    assert rows
    assert all(set(row) == FIELDS for row in rows)
    assert {row["worker_id"] for row in rows} == {0, 1}
    assert all(a["now_ms"] <= b["now_ms"] for a, b in pairwise(rows))
    # Every prompt token is prefilled once; the first output token comes with
    # the prefill iteration and the rest are decode iterations (no preemption).
    prompt_tokens = sum(64 + 32 * (i % 5) for i in range(120))
    output_tokens = sum(8 + i % 5 for i in range(120))
    assert sum(row["sum_prefill_tokens"] for row in rows) == prompt_tokens
    assert sum(row["num_decode_requests"] for row in rows) == output_tokens - 120


@pytest.mark.parametrize("turns_per_session", [1, 2])
def test_synthetic_replay_logs(tmp_path, turns_per_session):
    # One turn goes through the bindings' requests arm, two turns through the
    # workload arm.
    log = tmp_path / "workers.jsonl"
    report = run_synthetic_trace_replay(
        64,
        4,
        request_count=20,
        extra_engine_args=engine_args(),
        num_workers=2,
        arrival_interval_ms=5.0,
        turns_per_session=turns_per_session,
        fpm_log=log,
    )
    assert report["completed_requests"] == 20 * turns_per_session
    rows = lines(log)
    assert sum(row["num_decode_requests"] for row in rows) == 20 * turns_per_session * 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"replay_mode": "online"}, "fpm_log only supports replay_mode='offline'"),
        ({"replay_concurrency": 4}, "fpm_log does not support replay_concurrency"),
        (
            {
                "extra_engine_args": None,
                "prefill_engine_args": engine_args(worker_type="prefill"),
                "decode_engine_args": engine_args(worker_type="decode"),
            },
            "fpm_log only supports aggregated replay",
        ),
    ],
)
def test_synthetic_rejected_combinations(tmp_path, kwargs, message):
    kwargs.setdefault("extra_engine_args", engine_args())
    with pytest.raises(ValueError, match=message):
        run_synthetic_trace_replay(
            64,
            4,
            20,
            arrival_interval_ms=5.0,
            fpm_log=tmp_path / "workers.jsonl",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"replay_concurrency": 4}, "fpm_log does not support replay_concurrency"),
        ({"trace_format": "dynamo"}, "fpm_log only supports trace_format='mooncake'"),
        ({"replay_mode": "online"}, "fpm_log only supports replay_mode='offline'"),
    ],
)
def test_rejected_combinations(tmp_path, kwargs, message):
    trace = write_trace(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match=message):
        run_trace_replay(
            trace,
            extra_engine_args=engine_args(),
            trace_block_size=TRACE_BLOCK_SIZE,
            fpm_log=tmp_path / "workers.jsonl",
            **kwargs,
        )


def test_disaggregated_replay_is_rejected(tmp_path):
    trace = write_trace(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match="fpm_log only supports aggregated replay"):
        run_trace_replay(
            trace,
            prefill_engine_args=engine_args(),
            decode_engine_args=engine_args(),
            trace_block_size=TRACE_BLOCK_SIZE,
            fpm_log=tmp_path / "workers.jsonl",
        )


def test_unwritable_path_fails(tmp_path):
    trace = write_trace(tmp_path / "trace.jsonl")
    with pytest.raises(Exception, match="cannot create fpm_log"):
        run_trace_replay(
            trace,
            extra_engine_args=engine_args(),
            trace_block_size=TRACE_BLOCK_SIZE,
            fpm_log=tmp_path / "missing" / "workers.jsonl",
        )
