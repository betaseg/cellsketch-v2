"""Which entities get skeletons, and computing each object's only once.

Skeletonising is the most expensive measurement here, 103 s of a 2-minute object, and a
blob's skeleton is one branch the length of its diameter. ``--skeleton-entities`` names the
entities where branches, length and tortuosity are worth the cost.

The same skeletons feed the table metrics and the overlay geometry. Processors cannot pass
anything to each other, but they run in the same process and the same call per object
(``processing._process_memory_chunk``), so a cache holding one object's skeletons turns the
second computation into a lookup.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from pixel_patrol_anatomy.discovery import normalize_name
from pixel_patrol_anatomy.geometry import compute_skeletons

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


class _PerObjectCache:
    """One object's expensive intermediates, keyed however the caller asks.

    Validity is the object id *and* the identity of the array the values were computed from:
    object folder names are not guaranteed unique across groups, and a cache that answered on
    the name alone would hand one object another's results. The array is held by weak
    reference, so caching costs no memory beyond the cached values themselves.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._object_id: Optional[str] = None
        self._owner: Optional[weakref.ref] = None
        self._by_entity: Dict[Tuple[Any, ...], Any] = {}

    def get_or_compute(self, object_id: str, key: Tuple[Any, ...], array: np.ndarray, factory) -> Any:
        """The cached value for (object, key), or ``factory()`` stored under it."""
        owner = array.base if array.base is not None else array
        with self._lock:
            same_object = self._object_id == object_id and (
                self._owner is not None and self._owner() is owner
            )
            if not same_object:
                # A different object, or the same name over different data: drop the
                # previous entry rather than answering from it.
                self._object_id, self._by_entity = object_id, {}
                try:
                    self._owner = weakref.ref(owner)
                except TypeError:      # not weak-referenceable: no caching, still correct
                    self._owner = None
            cached = self._by_entity.get(key)
        if cached is not None:
            logger.debug("anatomy: reusing %s of %s", key, object_id)
            return cached

        computed = factory()
        with self._lock:
            if self._object_id == object_id and self._owner is not None and self._owner() is owner:
                self._by_entity[key] = computed
        return computed

    def clear(self) -> None:
        with self._lock:
            self._object_id, self._owner, self._by_entity = None, None, {}


# Process-local: each Dask worker handles one record at a time, so this holds the object
# currently being processed and nothing else.
CACHE = _PerObjectCache()


def skeletons_for(
    object_id: str,
    entity: str,
    labels: np.ndarray,
    sample_size: Sequence[float],
    max_voxels: Optional[int],
    num_threads: int,
) -> dict:
    """This entity's skeletons, computed once per object however many readers ask.

    TEASAR for a volume, a thinned medial axis for a plane; both answer branches, length
    and tortuosity, and share one cache.
    """
    return CACHE.get_or_compute(
        object_id, ("skeletons", entity, max_voxels), labels,
        lambda: compute_skeletons(
            labels, tuple(float(v) for v in sample_size),
            max_voxels=max_voxels, num_threads=num_threads,
        ),
    )


def label_metrics_for(object_id: str, entity: str, labels: np.ndarray,
                      sample_size: Sequence[float]) -> Dict[int, Dict[str, Any]]:
    """ITK's shape statistics for every instance of an entity, once per object.

    A --with-mesh run has two readers: the instance table, and the geometry rows that carry
    the same numbers beside each mesh. It is a full pass over the entity either time.
    """
    from pixel_patrol_anatomy.geometry import label_metrics

    return CACHE.get_or_compute(
        object_id, ("label_metrics", normalize_name(entity)), labels,
        lambda: label_metrics(labels, sample_size),
    )


def region_metrics_for(object_id: str, entity: str, volume: np.ndarray,
                       sample_size: Sequence[float]) -> Dict[str, float]:
    """Shape statistics for a whole-structure mask, once per object and entity.

    The same two readers the label entities have, and for a mask this is the single most
    expensive measurement in the object: the object mask is the largest structure there is.

    Keyed on the volume, not on the boolean derived from it. ``volume > 0`` is a fresh array
    every call, which would not merely miss - the cache would take it for a different object
    and evict everything already stored for this one.
    """
    from pixel_patrol_anatomy.geometry import region_metrics

    return CACHE.get_or_compute(
        object_id, ("region_metrics", normalize_name(entity)), volume,
        lambda: region_metrics(volume > 0, sample_size),
    )


def regions_for(object_id: str, entity: str, labels: np.ndarray):
    """``regionprops`` for an entity, once per object.

    The same two readers and the same full pass (find_objects) each time. Sharing the
    objects shares the properties they compute lazily too, so an area read for the instance
    table is not measured again for the geometry.
    """
    from skimage.measure import regionprops

    return CACHE.get_or_compute(
        object_id, ("regions", normalize_name(entity)), labels,
        lambda: regionprops(labels),
    )


def contacts_for(object_id: str, volumes, kinds, voxel_size_zyx, max_gap_um: float,
                 object_mask_name: str | None = None):
    """This object's contact edge list, computed once whether the table or the mesh CSV asks.

    Both want the same pairs at the same threshold, and finding them is seconds to minutes
    depending on the instance count.
    """
    from pixel_patrol_anatomy.contacts import pairwise_instance_gaps

    any_view = next(iter(volumes.values()))
    return CACHE.get_or_compute(
        object_id, ("contacts", round(float(max_gap_um), 6), object_mask_name), any_view,
        lambda: pairwise_instance_gaps(volumes, kinds, voxel_size_zyx, max_gap_um,
                                       object_mask_name),
    )
