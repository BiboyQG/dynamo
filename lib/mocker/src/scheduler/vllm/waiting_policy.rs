// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-pass reordering of the waiting queue by an installed policy.
//!
//! Before the admission loop of a pass, the core builds a [`WaitingQueueView`]
//! of every live waiting request and asks the policy for the order in which
//! to try them. The admission rules do not change: the first candidate that
//! does not fit stops the loop, nothing is preempted on a candidate's behalf,
//! and the order is not written back to the queue.

use std::sync::{Arc, Mutex};

use serde::Serialize;

/// One live waiting request, as the policy sees it. Sequence lengths and
/// prefix hits are in tokens; KV capacity is in blocks.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WaitingRequestView {
    /// Position among the live waiting requests in native order.
    pub queue_index: usize,
    /// Arrival order at this scheduler; the tie key.
    pub submit_seq: u64,
    /// Virtual time since the request first waited here; not reset by preemption.
    pub wait_ms: f64,
    /// Tokens this scheduler has processed since the request first waited here.
    pub age_tokens: u64,
    pub prompt_tokens: usize,
    /// Generated tokens kept across a preemption; zero for new requests.
    pub generated_tokens: usize,
    pub num_preemptions: usize,
    /// Prefix-cache hit on the known sequence (prompt plus retained generated
    /// tokens), clipped to its length minus one and then to whole blocks;
    /// under MTP one more block is recomputed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_prefix_tokens: Option<usize>,
    /// Blocks admission would allocate; `needed_blocks <= free_blocks` is the admission test.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub needed_blocks: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WaitingQueueView {
    pub now_ms: f64,
    /// Opaque key, one per scheduler instance.
    pub worker_id: u64,
    pub worker_role: &'static str,
    pub num_running: usize,
    pub max_num_seqs: usize,
    pub waiting_len: usize,
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    /// Blocks not held by running requests; includes evictable cached blocks.
    pub free_blocks: usize,
    pub requests: Vec<WaitingRequestView>,
}

pub(crate) trait WaitingQueuePolicy: Send {
    /// Whether the view must carry the prefix fields. A policy that needs them
    /// is skipped on queues deeper than [`PREFIX_HITS_WAITING_CAP`].
    fn uses_prefix_hits(&self) -> bool;

    /// The order in which this pass tries the requests, as a permutation of
    /// the indices of `view.requests`.
    fn order(&mut self, view: &WaitingQueueView) -> anyhow::Result<Vec<usize>>;
}

/// One policy object serves every worker of a replay; the replay is
/// single-threaded, so the lock never contends.
pub(crate) type WaitingPolicyHandle = Arc<Mutex<dyn WaitingQueuePolicy>>;

/// A prefix-aware policy is not called when more requests are waiting: the
/// prefix probe costs O(hit blocks) per request in a real engine.
pub(crate) const PREFIX_HITS_WAITING_CAP: usize = 128;

pub(super) fn validate_order(order: &[usize], n: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        order.len() == n,
        "waiting policy returned {} indices for {n} requests",
        order.len()
    );
    let mut seen = vec![false; n];
    for &index in order {
        anyhow::ensure!(
            index < n && !seen[index],
            "waiting policy returned an invalid or repeated index {index} for {n} requests"
        );
        seen[index] = true;
    }
    Ok(())
}
