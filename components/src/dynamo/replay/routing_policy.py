# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-scoring policies for fixed-worker, offline aggregated replay.

A fresh policy file is loaded once per replay. ``score(worker, ctx)`` returns a
number; lower wins. ``uses_prefix_hits=True`` adds cached-prefix and potential
prefill fields. Optional ``on_route(worker, ctx)`` runs once after assignment.
Inputs are detached snapshots. Only past/current router observations are exposed;
no actual future output length. Scores never advance the simulated clock.
"""

import copy
import importlib.util
import json
import math
import numbers
import os
import sys
import time
from types import ModuleType


def load(path):
    path = os.fspath(path)
    name = "routing_policy_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} is not a Python source file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ScoreWorkers:
    """Adapter called once per routing decision, including one-worker replays.

    Ties use the native replay-bench seeded reservoir rule over workers sorted
    by ID/rank. NaN is +inf; all +inf still selects a worker, never drops one.
    Exceptions abort the replay. Changing snapshots cannot modify simulator state.
    """

    def __init__(self, policy, decision_log=None):
        self.module = policy if isinstance(policy, ModuleType) else load(policy)
        self.score = self.module.score
        self.on_route = getattr(self.module, "on_route", None)
        self.uses_prefix_hits = bool(getattr(self.module, "uses_prefix_hits", False))
        self.decision_log = decision_log
        self.stats = dict(calls=0, scored_workers=0, seconds=0.0, assignments={})
        self._scores = []

    def scores(self, view):
        started = time.perf_counter()
        values = []
        for worker in view["workers"]:
            value = self.score(dict(worker), copy.deepcopy(view))
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(
                    f"score must return a number, got {type(value).__name__}"
                )
            values.append(math.inf if value != value else float(value))
        self.stats["calls"] += 1
        self.stats["scored_workers"] += len(values)
        self.stats["seconds"] += time.perf_counter() - started
        self._scores = values
        return values

    def committed(self, view, index):
        worker = view["workers"][index]
        key = f"{worker['worker_id']}:{worker['dp_rank']}"
        self.stats["assignments"][key] = self.stats["assignments"].get(key, 0) + 1
        if self.decision_log is not None:
            # JSON has no infinity: keep the score's explicit string representation.
            scores = [v if math.isfinite(v) else str(v) for v in self._scores]
            self.decision_log.write(
                json.dumps(
                    dict(
                        view,
                        scores=scores,
                        chosen_worker_id=worker["worker_id"],
                        chosen_dp_rank=worker["dp_rank"],
                    )
                )
                + "\n"
            )
        if self.on_route is not None:
            started = time.perf_counter()
            self.on_route(dict(worker), copy.deepcopy(view))
            self.stats["seconds"] += time.perf_counter() - started
