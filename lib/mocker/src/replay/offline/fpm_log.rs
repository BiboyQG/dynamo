// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-iteration forward-pass log for offline aggregated replay: one JSON line
//! per worker iteration, carrying the same fields a scaling policy sees in its
//! tick snapshots, plus the simulated time the iteration ended.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::common::protocols::ForwardPassSnapshot;

pub(super) struct FpmLog {
    writer: BufWriter<File>,
}

impl FpmLog {
    pub(super) fn create(path: &Path) -> anyhow::Result<Self> {
        let file = File::create(path)
            .map_err(|e| anyhow::anyhow!("cannot create fpm_log {}: {e}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// `now_ms` is the simulated time the iteration ended; `wall_time` inside
    /// the line is its duration in seconds, so it started at `now_ms - 1000 * wall_time`.
    pub(super) fn write(
        &mut self,
        now_ms: f64,
        worker_id: usize,
        fpm: &ForwardPassSnapshot,
    ) -> anyhow::Result<()> {
        let line = serde_json::json!({
            "now_ms": now_ms,
            "worker_id": worker_id,
            "dp_rank": fpm.dp_rank,
            "wall_time": fpm.wall_time_secs,
            "num_prefill_requests": fpm.num_prefill_requests,
            "sum_prefill_tokens": fpm.sum_prefill_tokens,
            "var_prefill_length": fpm.var_prefill_length,
            "sum_prefill_kv_tokens": fpm.sum_prefill_kv_tokens,
            "num_decode_requests": fpm.num_decode_requests,
            "sum_decode_kv_tokens": fpm.sum_decode_kv_tokens,
            "var_decode_kv_tokens": fpm.var_decode_kv_tokens,
            "num_queued_prefill": fpm.num_queued_prefill,
            "sum_queued_prefill_tokens": fpm.sum_queued_prefill_tokens,
            "var_queued_prefill_length": fpm.var_queued_prefill_length,
            "num_queued_decode": fpm.num_queued_decode,
            "sum_queued_decode_kv_tokens": fpm.sum_queued_decode_kv_tokens,
            "var_queued_decode_kv_tokens": fpm.var_queued_decode_kv_tokens,
        });
        serde_json::to_writer(&mut self.writer, &line)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    pub(super) fn flush(&mut self) -> anyhow::Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}
