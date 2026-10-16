// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An installed waiting-queue policy reaches every aggregated worker of an
//! offline replay, and the identity policy leaves the replay unchanged.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use tempfile::NamedTempFile;
use uuid::Uuid;

use super::scaling::{ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot};
use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, SglangArgs};
use crate::loadgen::{SessionTrace, Trace, TraceFileFormat, TurnTrace};
use crate::replay::{ReplayRouterMode, SlaThresholds, TraceSimulationReport};
use crate::scheduler::{WaitingPolicyHandle, WaitingQueuePolicy, WaitingQueueView};

/// Keeps the native order and records the workers it was asked about.
struct IdentityPolicy {
    workers_seen: Arc<Mutex<BTreeSet<u64>>>,
}

impl WaitingQueuePolicy for IdentityPolicy {
    fn uses_prefix_hits(&self) -> bool {
        true
    }

    fn order(&mut self, view: &WaitingQueueView) -> anyhow::Result<Vec<usize>> {
        self.workers_seen.lock().unwrap().insert(view.worker_id);
        Ok((0..view.requests.len()).collect())
    }
}

fn identity_policy() -> (WaitingPolicyHandle, Arc<Mutex<BTreeSet<u64>>>) {
    let workers_seen = Arc::new(Mutex::new(BTreeSet::new()));
    let policy = IdentityPolicy {
        workers_seen: Arc::clone(&workers_seen),
    };
    (Arc::new(Mutex::new(policy)), workers_seen)
}

/// Adds a second worker at the first scaling tick.
pub(super) struct AddOneWorker;

impl ReplayScalingPolicy for AddOneWorker {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn on_tick(&mut self, _: ReplayScalingSnapshot) -> anyhow::Result<ReplayScalingDecision> {
        Ok(ReplayScalingDecision {
            target_decode: Some(2),
            next_tick_ms: None,
            ..Default::default()
        })
    }
}

/// Two running requests per worker, so a burst queues on every worker and
/// the policy is consulted.
pub(super) fn vllm_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(64)
        .max_num_batched_tokens(Some(16))
        .max_num_seqs(Some(2))
        .enable_prefix_caching(true)
        .enable_chunked_prefill(true)
        .build()
        .unwrap()
}

/// `n` requests one millisecond apart; three prompt families give prefix reuse.
pub(super) fn burst(n: usize) -> Vec<DirectRequest> {
    (0..n)
        .map(|i| {
            let token = [1, 2, 3][i % 3];
            DirectRequest {
                tokens: vec![token; 8 + 4 * (i % 4)],
                max_output_tokens: 3,
                uuid: Some(Uuid::from_u128(i as u128 + 1)),
                arrival_timestamp_ms: Some(i as f64),
                ..Default::default()
            }
        })
        .collect()
}

fn replay(
    args: MockEngineArgs,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    requests: Vec<DirectRequest>,
    scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    waiting_policy: Option<WaitingPolicyHandle>,
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
        waiting_policy,
        None,
    )
}

/// Round to a nanosecond: the summary means are summed in uuid-hash order,
/// and file-loaded or workload requests get fresh uuids on every run without
/// replay-bench, so the last bits of a mean can differ between two runs.
fn round_floats(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Number(n) if n.is_f64() => {
            *value = serde_json::json!((n.as_f64().unwrap() * 1e6).round() / 1e6);
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(round_floats),
        serde_json::Value::Object(map) => map.values_mut().for_each(round_floats),
        _ => {}
    }
}

/// The summary without the replay wall time and the rates derived from it,
/// plus the per-request records.
pub(super) fn comparable(report: &TraceSimulationReport) -> (serde_json::Value, serde_json::Value) {
    let mut summary = serde_json::to_value(report).unwrap();
    round_floats(&mut summary);
    for key in [
        "wall_time_ms",
        "processed_tokens_per_s",
        "processed_output_tokens_per_s",
    ] {
        summary
            .as_object_mut()
            .unwrap()
            .remove(key)
            .expect("summary has the wall-time keys");
    }
    let mut records = serde_json::to_value(&report.per_request).unwrap();
    round_floats(&mut records);
    for record in records.as_array_mut().unwrap() {
        record.as_object_mut().unwrap().remove("uuid");
    }
    records.as_array_mut().unwrap().sort_by(|a, b| {
        a["arrival_time_ms"]
            .as_f64()
            .partial_cmp(&b["arrival_time_ms"].as_f64())
            .unwrap()
    });
    (summary, records)
}

/// `n` two-turn sessions one millisecond apart; the second turn extends the first.
pub(super) fn sessions(n: usize) -> Trace {
    Trace {
        block_size: 4,
        sessions: (0..n)
            .map(|i| {
                let base = i as u32 * 10;
                SessionTrace {
                    session_id: format!("session-{i}"),
                    first_arrival_timestamp_ms: Some(i as f64),
                    turns: vec![
                        TurnTrace {
                            input_length: 8,
                            max_output_tokens: 3,
                            hash_ids: vec![base + 1, base + 2],
                            delay_after_previous_ms: 0.0,
                            ..Default::default()
                        },
                        TurnTrace {
                            input_length: 12,
                            max_output_tokens: 3,
                            hash_ids: vec![base + 1, base + 2, base + 3],
                            delay_after_previous_ms: 0.0,
                            ..Default::default()
                        },
                    ],
                }
            })
            .collect(),
    }
}

/// `burst(n)` as a Mooncake trace file (trace block size 4), one session per
/// request; with `turns == 2` a second turn, one block longer, follows at once.
pub(super) fn mooncake_file(n: usize, turns: usize) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    for i in 0..n {
        let family = (i % 3) as u64 * 100;
        for turn in 0..turns {
            let input_length = 8 + 4 * (i % 4) + 4 * turn;
            let hash_ids: Vec<u64> = (0..input_length / 4).map(|b| family + b as u64).collect();
            let mut row = serde_json::json!({
                "session_id": format!("session-{i}"),
                "input_length": input_length,
                "output_length": 3,
                "hash_ids": hash_ids,
            });
            if turn == 0 {
                row["timestamp"] = serde_json::json!(i as f64);
            } else {
                row["delay"] = serde_json::json!(0.0);
            }
            writeln!(file, "{row}").unwrap();
        }
    }
    file
}

/// Records (submit_seq, now_ms, wait_ms) of every request it is shown.
struct RecordWaits(Arc<Mutex<Vec<(u64, f64, f64)>>>);

impl WaitingQueuePolicy for RecordWaits {
    fn uses_prefix_hits(&self) -> bool {
        false
    }

    fn order(&mut self, view: &WaitingQueueView) -> anyhow::Result<Vec<usize>> {
        let mut rows = self.0.lock().unwrap();
        for request in &view.requests {
            rows.push((request.submit_seq, view.now_ms, request.wait_ms));
        }
        Ok((0..view.requests.len()).collect())
    }
}

/// The driver passes the arrival time into receive: burst(n) arrives one
/// millisecond apart on one worker, so wait_ms is now_ms minus (submit_seq - 1).
#[test]
fn wait_ms_counts_from_the_arrival_the_driver_passes_in() {
    let rows = Arc::new(Mutex::new(Vec::new()));
    let policy: WaitingPolicyHandle = Arc::new(Mutex::new(RecordWaits(Arc::clone(&rows))));
    replay(
        vllm_args(),
        1,
        ReplayRouterMode::RoundRobin,
        burst(12),
        None,
        Some(policy),
    )
    .unwrap();
    let rows = rows.lock().unwrap();
    assert!(!rows.is_empty());
    for &(submit_seq, now_ms, wait_ms) in rows.iter() {
        assert_eq!(
            wait_ms,
            now_ms - (submit_seq as f64 - 1.0),
            "submit_seq {submit_seq} at {now_ms} ms"
        );
    }
}

#[test]
fn identity_policy_matches_native_order_on_one_worker() {
    // A policy keeps one worker on the multi-worker runtime, so the native
    // run uses that runtime too (SingleRuntime parity has its own tests).
    let (policy, workers_seen) = identity_policy();
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
    let ordered = replay(
        vllm_args(),
        1,
        ReplayRouterMode::RoundRobin,
        burst(12),
        None,
        Some(policy),
    )
    .unwrap();
    assert_eq!(native.request_counts.completed_requests, 12);
    assert_eq!(comparable(&native), comparable(&ordered));
    assert_eq!(*workers_seen.lock().unwrap(), BTreeSet::from([0]));
}

#[test]
fn identity_policy_matches_native_order_on_four_round_robin_workers() {
    let (policy, workers_seen) = identity_policy();
    let native = replay(
        vllm_args(),
        4,
        ReplayRouterMode::RoundRobin,
        burst(40),
        None,
        None,
    )
    .unwrap();
    let ordered = replay(
        vllm_args(),
        4,
        ReplayRouterMode::RoundRobin,
        burst(40),
        None,
        Some(policy),
    )
    .unwrap();
    assert_eq!(native.request_counts.completed_requests, 40);
    assert_eq!(comparable(&native), comparable(&ordered));
    assert_eq!(*workers_seen.lock().unwrap(), BTreeSet::from([0, 1, 2, 3]));
}

/// The KV router breaks placement ties with a seeded RNG only under
/// `replay-bench` (the feature the Python bindings are built with); without
/// it two native replays already differ.
#[cfg(feature = "replay-bench")]
#[test]
fn identity_policy_matches_native_order_on_four_kv_router_workers() {
    let (policy, workers_seen) = identity_policy();
    let native = replay(
        vllm_args(),
        4,
        ReplayRouterMode::KvRouter,
        burst(40),
        None,
        None,
    )
    .unwrap();
    let ordered = replay(
        vllm_args(),
        4,
        ReplayRouterMode::KvRouter,
        burst(40),
        None,
        Some(policy),
    )
    .unwrap();
    assert_eq!(native.request_counts.completed_requests, 40);
    assert_eq!(comparable(&native), comparable(&ordered));
    assert_eq!(*workers_seen.lock().unwrap(), BTreeSet::from([0, 1, 2, 3]));
}

#[test]
fn worker_added_by_scaling_gets_the_policy() {
    // With attention DP the added worker brings two schedulers, keys 2 and 3.
    let mut args = vllm_args();
    args.dp_size = 2;
    let (policy, workers_seen) = identity_policy();
    let report = replay(
        args,
        1,
        ReplayRouterMode::RoundRobin,
        burst(48),
        Some(Box::new(AddOneWorker)),
        Some(policy),
    )
    .unwrap();
    assert_eq!(report.request_counts.completed_requests, 48);
    assert_eq!(*workers_seen.lock().unwrap(), BTreeSet::from([0, 1, 2, 3]));
}

#[test]
fn attention_dp_ranks_each_get_the_policy() {
    // worker_id in the view is worker * dp_size + rank: four distinct keys.
    let mut args = vllm_args();
    args.dp_size = 2;
    let (policy, workers_seen) = identity_policy();
    let native = replay(
        args.clone(),
        2,
        ReplayRouterMode::RoundRobin,
        burst(24),
        None,
        None,
    )
    .unwrap();
    let ordered = replay(
        args,
        2,
        ReplayRouterMode::RoundRobin,
        burst(24),
        None,
        Some(policy),
    )
    .unwrap();
    assert_eq!(native.request_counts.completed_requests, 24);
    assert_eq!(comparable(&native), comparable(&ordered));
    assert_eq!(*workers_seen.lock().unwrap(), BTreeSet::from([0, 1, 2, 3]));
}

#[test]
fn requests_entrypoint_consults_the_policy() {
    // The bindings' path for single-turn requests; one worker must not fall
    // back to SingleRuntime, which has no policy.
    let (policy, workers_seen) = identity_policy();
    let native = super::entrypoints::simulate_trace_multi(
        vllm_args(),
        None,
        None,
        burst(12),
        1,
        1.0,
        ReplayRouterMode::RoundRobin,
        false,
        None,
        SlaThresholds::default(),
    )
    .unwrap();
    let ordered = crate::replay::simulate_trace_requests_with_router_mode_and_scaling_policy(
        vllm_args(),
        None,
        None,
        burst(12),
        1,
        1.0,
        ReplayRouterMode::RoundRobin,
        SlaThresholds::default(),
        None,
        Some(policy),
        None,
    )
    .unwrap();
    assert_eq!(native.request_counts.completed_requests, 12);
    assert_eq!(comparable(&native), comparable(&ordered));
    assert_eq!(*workers_seen.lock().unwrap(), BTreeSet::from([0]));
}

#[test]
fn workload_entrypoint_consults_the_policy() {
    // The bindings' path for synthetic multi-turn traces.
    let (policy, workers_seen) = identity_policy();
    let native = super::entrypoints::simulate_trace_workload_multi(
        vllm_args(),
        None,
        None,
        sessions(8),
        1,
        ReplayRouterMode::RoundRobin,
        false,
        true,
        true,
        None,
        SlaThresholds::default(),
    )
    .unwrap();
    let ordered =
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
            Some(policy),
            None,
        )
        .unwrap();
    assert_eq!(native.request_counts.completed_requests, 16);
    assert_eq!(comparable(&native), comparable(&ordered));
    assert_eq!(*workers_seen.lock().unwrap(), BTreeSet::from([0]));
}

#[test]
fn trace_file_entrypoint_consults_the_policy() {
    // The bindings' path for a Mooncake trace file; a two-turn file takes the
    // entrypoint's multi-turn branch.
    for turns in [1, 2] {
        let file = mooncake_file(24, turns);
        let run = |waiting_policy| {
            crate::replay::simulate_trace_file_with_router_mode_and_format_and_scaling_policy(
                vllm_args(),
                None,
                None,
                file.path(),
                4,
                2,
                1.0,
                ReplayRouterMode::RoundRobin,
                TraceFileFormat::Mooncake,
                0.0,
                0,
                true,
                None,
                SlaThresholds::default(),
                None,
                waiting_policy,
                None,
            )
            .unwrap()
        };
        let (policy, workers_seen) = identity_policy();
        let native = run(None);
        let ordered = run(Some(policy));
        assert_eq!(native.request_counts.completed_requests, 24 * turns);
        assert_eq!(comparable(&native), comparable(&ordered));
        assert_eq!(*workers_seen.lock().unwrap(), BTreeSet::from([0, 1]));
    }
}

/// Placement ties are random without replay-bench, so only the install is
/// checked here; the report comparison is the gated test above.
#[test]
fn kv_router_runtimes_install_the_policy() {
    let (policy, workers_seen) = identity_policy();
    let report = replay(
        vllm_args(),
        2,
        ReplayRouterMode::KvRouter,
        burst(24),
        None,
        Some(policy),
    )
    .unwrap();
    assert_eq!(report.request_counts.completed_requests, 24);
    assert!(!workers_seen.lock().unwrap().is_empty());

    let (policy, workers_seen) = identity_policy();
    let report = super::entrypoints::simulate_trace_workload_with_scaling_policy(
        vllm_args(),
        None,
        None,
        sessions(12),
        2,
        ReplayRouterMode::KvRouter,
        true,
        true,
        None,
        SlaThresholds::default(),
        None,
        Some(policy),
        None,
    )
    .unwrap();
    assert_eq!(report.request_counts.completed_requests, 24);
    assert!(!workers_seen.lock().unwrap().is_empty());
}

#[test]
fn sglang_engine_rejects_the_policy() {
    let args = MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .sglang(Some(SglangArgs {
            page_size: Some(4),
            chunked_prefill_size: Some(16),
            ..Default::default()
        }))
        .block_size(4)
        .num_gpu_blocks(64)
        .max_num_batched_tokens(Some(16))
        .max_num_seqs(Some(2))
        .build()
        .unwrap();
    let (policy, _) = identity_policy();
    let err = replay(
        args,
        1,
        ReplayRouterMode::RoundRobin,
        burst(4),
        None,
        Some(policy),
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("only supported by the vLLM scheduler"),
        "{err:#}"
    );
}

#[test]
fn agentic_and_delta_trace_formats_reject_the_policy() {
    for (format, message) in [
        (
            TraceFileFormat::AgenticMooncake,
            "waiting_policy replay only supports standard Mooncake traces",
        ),
        (
            TraceFileFormat::MooncakeDelta,
            "waiting_policy replay does not support mooncake-delta traces",
        ),
    ] {
        let (policy, _) = identity_policy();
        // Both checks run before the trace file is opened.
        let err =
            crate::replay::simulate_trace_file_with_router_mode_and_format_and_scaling_policy(
                vllm_args(),
                None,
                None,
                Path::new("/nonexistent/trace.jsonl"),
                4,
                1,
                1.0,
                ReplayRouterMode::RoundRobin,
                format,
                0.0,
                0,
                false,
                None,
                SlaThresholds::default(),
                None,
                Some(policy),
                None,
            )
            .unwrap_err();
        assert!(err.to_string().contains(message), "{format:?}: {err:#}");
    }
}
