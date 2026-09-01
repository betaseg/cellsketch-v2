"""Branches, length and tortuosity, read off a curve skeleton's graph.

These are the three metrics a filament is judged by — they drive their own violins, and
the gallery sorts by them — and all three come from the shape of the skeleton graph rather
than from the voxels. So the graph is what the tests hand over: a stand-in skeleton with
the vertices and edges of a known topology, which pins the counting rules exactly
(a straight strand is one branch, a Y is three, a closed loop is one) without depending on
what kimimaro happens to skeletonise a small synthetic volume into.

The tortuosity cases are geometric for the same reason: a straight branch is 1 by
definition, and a quarter circle is π/2 ÷ √2 whatever produced it.
"""

import numpy as np
import pytest

from pixel_patrol_anatomy.geometry import skeleton_graph_metrics


class FakeSkeleton:
    """The three things skeleton_graph_metrics reads off a kimimaro Skeleton."""

    def __init__(self, vertices, edges):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.edges = np.asarray(edges, dtype=np.uint32)

    def cable_length(self) -> float:
        if not len(self.edges):
            return 0.0
        a = self.vertices[self.edges[:, 0]]
        b = self.vertices[self.edges[:, 1]]
        return float(np.sum(np.linalg.norm(b - a, axis=1)))


def _chain(n: int, spacing: float = 1.0) -> FakeSkeleton:
    vertices = [(i * spacing, 0.0, 0.0) for i in range(n)]
    return FakeSkeleton(vertices, [(i, i + 1) for i in range(n - 1)])


# ── how many branches ─────────────────────────────────────────────────────────

def test_a_straight_strand_is_one_branch():
    metrics = skeleton_graph_metrics(_chain(5))

    assert metrics["branches"] == 1
    assert metrics["length_um"] == pytest.approx(4.0)


def test_a_y_is_three_branches():
    #  0-1-2 <- trunk, then two arms off the junction at 2
    vertices = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 1, 0), (3, -1, 0)]
    edges = [(0, 1), (1, 2), (2, 3), (2, 4)]

    metrics = skeleton_graph_metrics(FakeSkeleton(vertices, edges))

    # The junction ends the trunk and starts both arms: a chain between nodes of degree
    # != 2, counted three times, not one strand with a kink in it.
    assert metrics["branches"] == 3


def test_a_closed_loop_is_one_branch():
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    metrics = skeleton_graph_metrics(FakeSkeleton(vertices, edges))

    # Every vertex has degree 2, so a loop has no endpoint and no junction to start from.
    # Counting only from breakpoints would score it 0 branches - a ring of ER reported as
    # having no skeleton at all.
    assert metrics["branches"] == 1
    assert metrics["length_um"] == pytest.approx(4.0)


def test_a_skeleton_that_falls_into_two_pieces_counts_both():
    left = _chain(3)
    right = _chain(3)
    vertices = np.vstack([left.vertices, right.vertices + np.array([0, 10, 0])])
    edges = np.vstack([left.edges, right.edges + 3])

    metrics = skeleton_graph_metrics(FakeSkeleton(vertices, edges))

    assert metrics["branches"] == 2


def test_an_instance_with_no_skeleton_measures_zero_and_not_nan_length():
    for empty in (None, FakeSkeleton([], []), FakeSkeleton([(0, 0, 0)], [])):
        metrics = skeleton_graph_metrics(empty)

        # Nothing to measure is 0 branches of 0 length; tortuosity has no meaning at all,
        # and NULL is what the processors write for that.
        assert metrics["branches"] == 0
        assert metrics["length_um"] == 0.0
        assert np.isnan(metrics["tortuosity"])


# ── how bent ──────────────────────────────────────────────────────────────────

def test_a_straight_branch_has_tortuosity_one():
    assert skeleton_graph_metrics(_chain(9))["tortuosity"] == pytest.approx(1.0, abs=1e-6)


def test_a_curved_branch_reports_its_arc_over_its_chord():
    angles = np.linspace(0, np.pi / 2, 40)
    vertices = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    edges = [(i, i + 1) for i in range(len(angles) - 1)]

    tortuosity = skeleton_graph_metrics(FakeSkeleton(vertices, edges))["tortuosity"]

    # A quarter circle of radius 1: arc π/2 over chord √2, ~1.11. Smoothing pulls a real
    # skeleton's voxel staircase out before this is measured, and on a genuinely curved
    # path it barely moves the answer - which is the property that makes the number mean
    # "bent", rather than "voxelised".
    assert tortuosity == pytest.approx((np.pi / 2) / np.sqrt(2), abs=0.02)


def test_a_staircase_reads_as_straighter_than_its_voxels():
    # The 1-voxel staircase a voxelised diagonal carries: 45° of steps, arc 2x the chord
    # before smoothing.
    vertices = []
    for i in range(12):
        vertices += [(i, i, 0), (i + 1, i, 0)]
    edges = [(i, i + 1) for i in range(len(vertices) - 1)]

    tortuosity = skeleton_graph_metrics(FakeSkeleton(vertices, edges))["tortuosity"]

    # Unsmoothed this reads ~1.41 (the staircase's arc over the diagonal's chord), which
    # would make every straight-but-oblique filament look bent.
    assert 1.0 <= tortuosity < 1.2
