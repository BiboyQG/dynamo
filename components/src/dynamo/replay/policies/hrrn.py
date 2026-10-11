# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Highest response ratio next: (waiting + service) / service, with waiting
measured in tokens the scheduler processed meanwhile and service as the
tokens left to compute after the prefix-cache hit."""

uses_prefix_hits = True


def score(request, ctx):
    known = request["prompt_tokens"] + request["generated_tokens"]
    service = max(known - request["cached_prefix_tokens"], 1)
    return -(request["age_tokens"] + service) / service
