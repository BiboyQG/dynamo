// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The per-iteration FPM log: one line per worker iteration on every
//! aggregated runtime, an unchanged report, and the waiting policy's
//! format restrictions.

use std::collections::BTreeSet;
use std::path::Path;

use tempfile::NamedTempFile;

use super::waiting_policy_tests::{
    AddOneWorker, burst, comparable, mooncake_file, sessions, vllm_args,
};
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::TraceFileFormat;
use crate::replay::{ReplayRouterMode, SlaThresholds, TraceSimulationReport};

const FIELDS: [&str; 17] = [
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
];

fn replay(
    args: MockEngineArgs,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    requests: Vec<DirectRequest>,
    scaling_policy: Option<Box<dyn super::scaling::ReplayScalingPolicy>>,
    fpm_log: Option<&Path>,
) -> anyhow::Result<TraceSimulationReport> {
    super::entrypoints::simulate_trace_with_scaling_policy(
        args,
        None,
        None,
        requests,
        num_workers,
        1.0,
        router_mode,
        true,
        None,
        SlaThresholds::default(),
        scaling_policy,
        None,
        fpm_log.map(Path::to_path_buf),
    )
}

fn lines(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn workers(lines: &[serde_json::Value]) -> BTreeSet<u64> {
    lines
        .iter()
        .map(|l| l["worker_id"].as_u64().unwrap())
        .collect()
}

fn sum(lines: &[serde_json::Value], key: &str) -> u64 {
    lines.iter().map(|l| l[key].as_u64().unwrap()).sum()
}

/// No prefix caching, so every prompt token is prefilled exactly once.
fn args_without_prefix_caching() -> MockEngineArgs {
    let mut args = vllm_args();
    args.enable_prefix_caching = false;
    args
}

#[test]
fn one_line_per_iteration_and_the_report_is_unchanged() {
    let native = replay(
        args_without_prefix_caching(),
        4,
        ReplayRouterMode::RoundRobin,
        burst(40),
        None,
        None,
    )
    .unwrap();
    let log = NamedTempFile::new().unwrap();
    let logged = replay(
        args_without_prefix_caching(),
        4,
        ReplayRouterMode::RoundRobin,
        burst(40),
        None,
        Some(log.path()),
    )
    .unwrap();
    assert_eq!(native.request_counts.completed_requests, 40);
    assert_eq!(comparable(&native), comparable(&logged));

    let lines = lines(log.path());
    assert!(!lines.is_empty());
    let fields: BTreeSet<&str> = FIELDS.into_iter().collect();
    for line in &lines {
        let keys: BTreeSet<&str> = line
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, fields);
        assert!(line["wall_time"].as_f64().unwrap() >= 0.0);
        assert_eq!(line["dp_rank"], 0);
    }
    assert_eq!(workers(&lines), BTreeSet::from([0, 1, 2, 3]));
    assert!(
        lines
            .windows(2)
            .all(|w| w[0]["now_ms"].as_f64().unwrap() <= w[1]["now_ms"].as_f64().unwrap()),
        "lines are written in simulated-time order"
    );
    // Three output tokens per request: the first comes with the prefill
    // iteration, the other two are decode iterations.
    assert_eq!(sum(&lines, "num_decode_requests"), 40 * 2);
    let prompt_tokens: u64 = (0..40).map(|i| 8 + 4 * (i % 4)).sum();
    assert_eq!(sum(&lines, "sum_prefill_tokens"), prompt_tokens);
}

#[test]
fn one_worker_runs_on_the_logging_runtime() {
    // SingleRuntime has no log, so a log keeps one worker on the multi-worker
    // runtime, like a waiting policy does.
    let native = super::entrypoints::simulate_trace_multi(
        vllm_args(),
        None,
        None,
        burst(12),
        1,
        1.0,
        ReplayRouterMode::RoundRobin,
        true,
        None,
        SlaThresholds::default(),
    )
    .unwrap();
    let log = NamedTempFile::new().unwrap();
    let logged = replay(
        vllm_args(),
        1,
        ReplayRouterMode::RoundRobin,
        burst(12),
        None,
        Some(log.path()),
    )
    .unwrap();
    assert_eq!(comparable(&native), comparable(&logged));
    let lines = lines(log.path());
    assert_eq!(workers(&lines), BTreeSet::from([0]));
    assert_eq!(sum(&lines, "num_decode_requests"), 12 * 2);
}

#[test]
fn a_worker_added_by_scaling_is_logged() {
    let log = NamedTempFile::new().unwrap();
    let report = replay(
        vllm_args(),
        1,
        ReplayRouterMode::RoundRobin,
        burst(24),
        Some(Box::new(AddOneWorker)),
        Some(log.path()),
    )
    .unwrap();
    assert_eq!(report.request_counts.completed_requests, 24);
    assert_eq!(workers(&lines(log.path())), BTreeSet::from([0, 1]));
}

#[test]
fn attention_dp_ranks_are_logged_with_their_worker() {
    let mut args = vllm_args();
    args.dp_size = 2;
    let log = NamedTempFile::new().unwrap();
    replay(
        args,
        2,
        ReplayRouterMode::RoundRobin,
        burst(24),
        None,
        Some(log.path()),
    )
    .unwrap();
    let ranks: BTreeSet<(u64, u64)> = lines(log.path())
        .iter()
        .map(|l| {
            (
                l["worker_id"].as_u64().unwrap(),
                l["dp_rank"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(ranks, BTreeSet::from([(0, 0), (0, 1), (1, 0), (1, 1)]));
}

#[test]
fn kv_router_runtime_logs() {
    let log = NamedTempFile::new().unwrap();
    let report = replay(
        vllm_args(),
        2,
        ReplayRouterMode::KvRouter,
        burst(24),
        None,
        Some(log.path()),
    )
    .unwrap();
    assert_eq!(report.request_counts.completed_requests, 24);
    assert_eq!(sum(&lines(log.path()), "num_decode_requests"), 24 * 2);
}

#[test]
fn requests_and_workload_entrypoints_log() {
    // The bindings' paths for single-turn requests and synthetic sessions.
    let log = NamedTempFile::new().unwrap();
    crate::replay::simulate_trace_requests_with_router_mode_and_scaling_policy(
        vllm_args(),
        None,
        None,
        burst(12),
        1,
        1.0,
        ReplayRouterMode::RoundRobin,
        SlaThresholds::default(),
        None,
        None,
        Some(log.path().to_path_buf()),
    )
    .unwrap();
    assert_eq!(sum(&lines(log.path()), "num_decode_requests"), 12 * 2);

    let log = NamedTempFile::new().unwrap();
    let report =
        crate::replay::simulate_trace_workload_with_router_mode_and_options_and_scaling_policy(
            vllm_args(),
            None,
            None,
            sessions(8),
            1,
            ReplayRouterMode::RoundRobin,
            true,
            None,
            SlaThresholds::default(),
            None,
            None,
            Some(log.path().to_path_buf()),
        )
        .unwrap();
    assert_eq!(report.request_counts.completed_requests, 16);
    assert_eq!(sum(&lines(log.path()), "num_decode_requests"), 16 * 2);
}

#[test]
fn trace_file_entrypoint_logs_and_rejects_other_formats() {
    let file = mooncake_file(24);
    let run = |format, fpm_log| {
        crate::replay::simulate_trace_file_with_router_mode_and_format_and_scaling_policy(
            vllm_args(),
            None,
            None,
            file.path(),
            4,
            2,
            1.0,
            ReplayRouterMode::RoundRobin,
            format,
            0.0,
            0,
            true,
            None,
            SlaThresholds::default(),
            None,
            None,
            fpm_log,
        )
    };
    let log = NamedTempFile::new().unwrap();
    let report = run(TraceFileFormat::Mooncake, Some(log.path().to_path_buf())).unwrap();
    assert_eq!(report.request_counts.completed_requests, 24);
    assert_eq!(workers(&lines(log.path())), BTreeSet::from([0, 1]));

    for (format, message) in [
        (
            TraceFileFormat::AgenticMooncake,
            "fpm_log replay only supports standard Mooncake traces",
        ),
        (
            TraceFileFormat::MooncakeDelta,
            "fpm_log replay does not support mooncake-delta traces",
        ),
    ] {
        let log = NamedTempFile::new().unwrap();
        let err = run(format, Some(log.path().to_path_buf())).unwrap_err();
        assert!(err.to_string().contains(message), "{format:?}: {err:#}");
    }
}

#[test]
fn an_unwritable_path_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing").join("workers.jsonl");
    let err = replay(
        vllm_args(),
        1,
        ReplayRouterMode::RoundRobin,
        burst(4),
        None,
        Some(&path),
    )
    .unwrap_err();
    assert!(err.to_string().contains("cannot create fpm_log"), "{err:#}");
}
