# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native order: preempted requests first, then arrival order."""

uses_prefix_hits = False


def score(request, ctx):
    return float(request["queue_index"])
