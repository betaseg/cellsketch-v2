"""Which entities get curve skeletons, and computing each cell's only once.

Skeletonising is the most expensive measurement here - 103 s of a 2-minute cell - and
most of it is usually wasted: a blob's skeleton is one branch the length of its diameter,
so entities like secretory granules pay for a number that says nothing. Hence
``--skeleton-entities``, which names the entities where branches, length and tortuosity
are worth having (filaments: mitochondria, ER).

The same skeletons feed two places - the metrics in the table and the overlay geometry in
report_meshes.csv - and processors cannot pass anything to each other. They do, however,
run in the same worker process and in the same call for a given cell
(``processing._process_memory_chunk`` loops over the processors), so a cache that holds
exactly one cell's skeletons turns the second computation into a lookup.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from pixel_patrol_cellsketch.discovery import normalize_name
from pixel_patrol_cellsketch.geometry import compute_curve_skeletons

logger = logging.getLogger(__name__)

# All entities unless a set is given; an empty set means none at all.
EntityFilter = Optional[FrozenSet[str]]


def wants_skeletons(entity: str, allowed: EntityFilter) -> bool:
    """Whether this entity is one the user asked to skeletonise.

    Names are compared normalised, so ``--skeleton-entities ER`` matches the entity
    discovered from ``sample_ER_label.tif`` as ``er``.
    """
    if allowed is None:
        return True
    return normalize_name(entity) in {normalize_name(name) for name in allowed}


def parse_entity_filter(raw: Optional[str], none: bool = False) -> EntityFilter:
    """A comma-separated list into an entity filter; None means every entity."""
    if none:
        return frozenset()
    if raw is None or not raw.strip():
        return None
    return frozenset(part for part in (p.strip() for p in raw.split(",")) if part)


class _PerCellCache:
    """One cell's expensive intermediates, keyed however the caller asks.

    Validity is the cell id *and* the identity of the array the values were computed from:
    cell folder names are not guaranteed unique across groups, and a cache that answered on
    the name alone would hand one cell another's results. The array is held by weak
    reference, so caching costs no memory beyond the cached values themselves.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cell_id: Optional[str] = None
        self._owner: Optional[weakref.ref] = None
        self._by_entity: Dict[Tuple[Any, ...], Any] = {}

    def get_or_compute(self, cell_id: str, key: Tuple[Any, ...], array: np.ndarray, factory) -> Any:
        """The cached value for (cell, key), or ``factory()`` stored under it."""
        owner = array.base if array.base is not None else array
        with self._lock:
            same_cell = self._cell_id == cell_id and (
                self._owner is not None and self._owner() is owner
            )
            if not same_cell:
                # A different cell, or the same name over different data: drop the
                # previous entry rather than answering from it.
                self._cell_id, self._by_entity = cell_id, {}
                try:
                    self._owner = weakref.ref(owner)
                except TypeError:      # not weak-referenceable: no caching, still correct
                    self._owner = None
            cached = self._by_entity.get(key)
        if cached is not None:
            logger.debug("cellsketch: reusing %s of %s", key, cell_id)
            return cached

        computed = factory()
        with self._lock:
            if self._cell_id == cell_id and self._owner is not None and self._owner() is owner:
                self._by_entity[key] = computed
        return computed

    def clear(self) -> None:
        with self._lock:
            self._cell_id, self._owner, self._by_entity = None, None, {}


# Process-local: each Dask worker handles one record at a time, so this holds the cell
# currently being processed and nothing else.
CACHE = _PerCellCache()


def skeletons_for(
    cell_id: str,
    entity: str,
    labels: np.ndarray,
    voxel_size_zyx: Sequence[float],
    max_voxels: Optional[int],
    num_threads: int,
) -> dict:
    """This entity's curve skeletons, computed once per cell however many readers ask."""
    return CACHE.get_or_compute(
        cell_id, ("skeletons", entity, max_voxels), labels,
        lambda: compute_curve_skeletons(
            labels, tuple(float(v) for v in voxel_size_zyx),
            max_voxels=max_voxels, num_threads=num_threads,
        ),
    )


def contacts_for(cell_id: str, volumes, kinds, voxel_size_zyx, max_gap_um: float):
    """This cell's contact edge list, computed once whether the table or the mesh CSV asks.

    Both want the same pairs at the same threshold, and finding them is seconds to minutes
    depending on the instance count.
    """
    from pixel_patrol_cellsketch.contacts import pairwise_instance_gaps

    any_view = next(iter(volumes.values()))
    return CACHE.get_or_compute(
        cell_id, ("contacts", round(float(max_gap_um), 6)), any_view,
        lambda: pairwise_instance_gaps(volumes, kinds, voxel_size_zyx, max_gap_um),
    )
