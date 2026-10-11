# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shortest prefill first: fewest tokens left to compute after the prefix-cache
hit; preempted requests go back first in their native order."""

uses_prefix_hits = True


def score(request, ctx):
    if request["num_preemptions"] > 0:
        # Back first, most recently preempted first: the native order. Keys
        # here are <= -1; a new request's key is never below -1.
        return request["queue_index"] - ctx["waiting_len"]
    known = request["prompt_tokens"] + request["generated_tokens"]
    return float(known - request["cached_prefix_tokens"])
