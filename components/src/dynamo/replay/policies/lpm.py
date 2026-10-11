# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Longest prefix match first; preempted requests go back first in their native order."""

uses_prefix_hits = True


def score(request, ctx):
    if request["num_preemptions"] > 0:
        # Back first, most recently preempted first: the native order. Keys
        # here are <= -1; a new request's key is never below -1.
        return request["queue_index"] - ctx["waiting_len"]
    # Scaled into (-1, 0] so the preempted keys stay ahead.
    capacity = ctx["num_gpu_blocks"] * ctx["block_size"]
    return -request["cached_prefix_tokens"] / capacity
