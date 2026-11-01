"""RoboDojo layout selection: episode_idx picks the counted layout, spares are split by shard."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from vla_eval.benchmarks.robodojo.benchmark import RoboDojoBenchmark


class _FakeEnv:
    """Single-env EvalEnv stand-in whose reset fails on the ``broken`` layouts."""

    def __init__(self, total: int, broken: set[int]) -> None:
        """``total`` layouts, of which ``broken`` raise on reset."""
        self.seed_manager = SimpleNamespace(seed_list=list(range(total)))
        self._broken = broken
        self.reset_calls: list[int] = []
        self.closes = 0
        self._layout = -1

    def close(self) -> None:
        """Count the close."""
        self.closes += 1

    def reset(self, seed: list[int]) -> None:
        """Reset to layout ``seed[0]``; raises when the layout is broken."""
        self.reset_calls.append(seed[0])
        if seed[0] in self._broken:
            raise RuntimeError("scene unstable")
        self._layout = seed[0]

    def run_reward(self) -> None:
        """No reward terms to update."""
        pass

    def get_obs(self) -> dict[str, Any]:
        """An observation tagged with the current layout."""
        return {"vision": {}, "layout": self._layout}


def _bench(total: int = 10, eval_num: int = 6, broken: set[int] | None = None) -> tuple[RoboDojoBenchmark, _FakeEnv]:
    """A RoboDojo benchmark over :class:`_FakeEnv` with ``eval_num`` scored layouts."""
    bench = RoboDojoBenchmark()
    env = _FakeEnv(total, broken or set())
    bench._env = env
    bench._current_task_name = "make_toast"
    bench._native_eval_num = eval_num
    bench._env_thread = threading.get_ident()  # as if _ensure_env built it on this thread
    return bench, env


def _run(bench: RoboDojoBenchmark, episodes: list[int], shard: dict[str, int] | None = None) -> list[int]:
    """Reset each episode in turn; return the layout each one landed on."""
    out = []
    for ep in episodes:
        bench.reset({"name": "make_toast", "episode_idx": ep, **(shard or {})})
        out.append(bench._current_layout_id)
    return out


def test_episode_idx_selects_layout() -> None:
    """Each episode runs on the layout of the same index."""
    bench, _ = _bench()
    assert _run(bench, [3, 0, 5]) == [3, 0, 5]


def test_single_shard_spares_match_upstream_set() -> None:
    """A single shard scores the same layouts as upstream."""
    # Upstream consumes 0..9 in order skipping 2 -> counts {0,1,3,4,5,6}.
    bench, _ = _bench(broken={2})
    assert sorted(_run(bench, list(range(6)))) == [0, 1, 3, 4, 5, 6]


def test_shards_never_share_layouts() -> None:
    """Shards draw spares from disjoint subsets."""
    # Spares are 6..9: shard 0 owns 6/8, shard 1 owns 7/9.
    shard0, _ = _bench(broken={0, 3, 6})
    shard1, _ = _bench(broken={0, 3, 6})
    got0 = _run(shard0, [0, 2], {"shard_id": 0, "num_shards": 2})
    got1 = _run(shard1, [1, 3, 5], {"shard_id": 1, "num_shards": 2})
    assert got0 == [8, 2]
    assert got1 == [1, 7, 5]
    assert set(got0).isdisjoint(got1)


def test_spare_exhaustion_raises() -> None:
    """A shard that runs out of spares raises."""
    bench, _ = _bench(total=8, broken={0, 6})
    with pytest.raises(RuntimeError, match="exhausted"):
        _run(bench, [0], {"shard_id": 0, "num_shards": 2})


def test_episode_idx_outside_counted_layouts_rejected() -> None:
    """An episode beyond the scored layouts is rejected."""
    bench, _ = _bench(eval_num=6)
    with pytest.raises(ValueError, match="episode_idx 6"):
        _run(bench, [6])


def test_consecutive_failures_capped() -> None:
    """The fallback stops after 5 failed layouts in a row instead of draining all spares."""
    bench, env = _bench(total=40, eval_num=25, broken=set(range(40)))
    with pytest.raises(RuntimeError, match="5 layouts in a row failed"):
        _run(bench, [0])
    assert len(env.reset_calls) == 5


def test_used_scene_closed_before_each_later_reset() -> None:
    """Every reset on a used scene closes it first."""
    # Upstream main.py closes between batches; a reset on a used scene stacks render products.
    bench, env = _bench(broken={1})
    _run(bench, [0])
    assert env.closes == 0
    _run(bench, [1, 2])  # layout 1 fails and falls back to spare 6
    assert env.reset_calls == [0, 1, 6, 2]
    assert env.closes == 3


def test_used_scene_never_closed_off_the_env_thread() -> None:
    """Off the env's own thread the scene is never closed."""
    # close() off the thread that built the env deadlocks in Kit; fall back to the soft reset.
    bench, env = _bench()
    bench._env_thread = -1
    _run(bench, [0, 1])
    assert env.closes == 0
