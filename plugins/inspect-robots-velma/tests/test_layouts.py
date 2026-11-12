"""Hidden-layout variations must not reveal the answer in policy instructions."""

import pytest

from inspect_robots_velma.task import velma_stack_three_task


@pytest.mark.parametrize("layout", ["shifted", "permuted", "random"])
def test_layout_changes_metadata_but_keeps_policy_input_identical(layout: str) -> None:
    """Only evaluator setup changes; nominal seeds and policy instructions stay fixed."""
    original = velma_stack_three_task().scenes[0]
    changed = velma_stack_three_task(layout=layout, layout_seed=17).scenes[0]
    assert changed.instruction == original.instruction
    assert any(
        changed.metadata[key] != original.metadata[key]
        for key in ("cube_x", "cube_y", "cube2_x", "cube2_y")
    )
    assert changed.target == original.target


def test_random_layout_is_reproducible_and_preserves_nonoverlap() -> None:
    """Rigid translation and permutation preserve cube spacing on the support pad."""
    for seed in range(20):
        a = velma_stack_three_task(layout="random", layout_seed=seed).scenes[0]
        b = velma_stack_three_task(layout="random", layout_seed=seed).scenes[0]
        assert a.metadata == b.metadata
        points = [
            (float(a.metadata[f"{name}_x"]), float(a.metadata[f"{name}_y"]))
            for name in ("cube", "cube2", "cube3")
        ]
        for i, p in enumerate(points):
            assert 0.025 < p[0] < 0.175 and -0.345 < p[1] < -0.195
            for q in points[i + 1 :]:
                assert max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 0.03 - 1e-12
