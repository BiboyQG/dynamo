// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-time snapshots for custom offline worker selection. No future trace data.
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Serialize)]
pub struct RoutingWorkerView {
    pub worker_id: u64,
    pub dp_rank: u32,
    pub worker_role: &'static str,
    pub active_requests: usize,
    pub active_prefill_tokens: usize,
    pub active_decode_blocks: usize,
    pub potential_decode_blocks: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_prefix_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub potential_prefill_tokens: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoutingRequestView {
    pub request_seq: u64,
    pub arrival_ms: f64,
    pub prompt_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoutingView {
    pub now_ms: f64,
    pub route_seq: u64,
    pub num_workers: usize,
    pub workers: Vec<RoutingWorkerView>,
    pub request: RoutingRequestView,
    pub block_size: usize,
    pub max_num_seqs: Option<usize>,
    pub max_num_batched_tokens: Option<usize>,
    pub num_gpu_blocks: usize,
    pub sla_ttft_ms: Option<f64>,
    pub sla_itl_ms: Option<f64>,
    pub sla_e2e_ms: Option<f64>,
}

/// Python's adapter scores all workers; selection and assignment stay native.
pub trait ReplayRoutingPolicy: Send {
    fn uses_prefix_hits(&self) -> bool;
    fn scores(&mut self, view: &RoutingView) -> anyhow::Result<Vec<f64>>;
    fn on_route(&mut self, view: &RoutingView, chosen_index: usize) -> anyhow::Result<()>;
}

pub type RoutingPolicyHandle = Arc<Mutex<dyn ReplayRoutingPolicy>>;
