"""Contact processor: which instances of an object touch, and how closely.

A contact is a property of an instance *pair*, so unlike morphology it belongs to no
single entity: this is a MEMORY processor, it sees every channel of the object at once,
and its output lands on the object's ``obs_level=0`` row as a five-column edge list.
Widgets unnest it in SQL, where parallel unnest() calls in one SELECT stay row-aligned,
and apply the gap threshold interactively:

    SELECT object_id, unnest(contact_entity_a) AS entity_a,
                    unnest(contact_entity_b) AS entity_b,
                    unnest(contact_gap_um)   AS gap_um
    FROM pp_data WHERE obs_level = 0

Exclude it with ``--processors-exclude anatomy-contacts`` (the equivalent of
the contact processor being excluded from the run).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec

from pixel_patrol_anatomy.config import AnatomyConfig
from pixel_patrol_anatomy.plugins.loaders.object_loader import OBJECT_KIND
from pixel_patrol_anatomy.spatial import voxel_size
from pixel_patrol_anatomy.plugins.processors.instances import channel_view
from pixel_patrol_anatomy.skeletons import contacts_for

logger = logging.getLogger(__name__)

_COLUMNS: Dict[str, Any] = {
    "contact_count": np.int64,
    "contact_entity_a": list,
    "contact_label_a": list,
    "contact_entity_b": list,
    "contact_label_b": list,
    "contact_gap_um": list,
}

_DESCRIPTIONS: Dict[str, str] = {
    "contact_count": "Number of instance pairs within the recorded gap threshold.",
    "contact_entity_a": "Entity of the first instance in each pair, in edge-list order shared by all contact_* columns.",
    "contact_label_a": "Label id of the first instance, null when it is a whole-structure mask.",
    "contact_entity_b": "Entity of the second instance in each pair.",
    "contact_label_b": "Label id of the second instance, null when it is a whole-structure mask.",
    "contact_gap_um": "Surface-to-surface gap of each pair in µm, measured in whole voxel steps, so instances sharing a face read one voxel step rather than zero.",
}


class ContactsProcessor:
    """Pairwise surface-to-surface gaps between the instances of one object."""

    NAME = "anatomy-contacts"
    DESCRIPTION = (
        "Records which instances of an object lie within a gap threshold of each other, as an "
        "edge list of instance pairs with their surface-to-surface gap in µm. The object mask "
        "is excluded: it encloses everything, so its proximity is a distance, not a contact."
    )

    CHUNK_KIND = ChunkKind.MEMORY
    # Y and X, not Z: a 2D object is a CYX record.
    INPUT = RecordSpec(axes={"C", "Y", "X"}, kinds={OBJECT_KIND})
    OUTPUT = "features"

    OUTPUT_SCHEMA: Dict[str, Any] = dict(_COLUMNS)
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = dict(_DESCRIPTIONS)

    def __init__(self) -> None:
        self._config = AnatomyConfig.from_env()

    def run_chunk(self, record: Record) -> Dict[str, Any]:
        meta = record.meta
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        dim_order = record.dim_order

        names = list(meta.get("channel_names") or [])
        kinds = list(meta.get("entity_kinds") or [])
        expected_zyx = [int(v) for v in (meta.get("object_shape") or [])]
        c_axis = dim_order.index("C")
        spatial = [s for i, s in enumerate(arr.shape) if i != c_axis]

        # Pairs span entities and the whole region, so a fragment under-reports contacts.
        if arr.shape[c_axis] != len(names) or (expected_zyx and spatial != expected_zyx):
            raise ValueError(
                f"object arrived as a {arr.shape} fragment of {len(names)}×{tuple(expected_zyx)}: "
                "an object is measured whole, so a fragment means the caller split it"
            )

        # Views, not copies: np.take would duplicate every entity volume.
        volumes = {name: channel_view(arr, c_axis, i) for i, name in enumerate(names)}
        kinds_by_name = dict(zip(names, kinds))
        sample_size = voxel_size(meta, record.dim_order)

        contacts = contacts_for(
            str(meta.get("object_id") or "object"), volumes, kinds_by_name, sample_size,
            self._config.contact_max_um, meta.get("object_mask_name"),
        )
        logger.info(
            "anatomy: %s: %d instance pairs within %.3g µm",
            meta.get("object_id"), len(contacts), self._config.contact_max_um,
        )
        if not contacts:
            # None, not []: an empty list types the column List(Null), which clashes with
            # the List(Varchar)/List(Double) written for objects that touch.
            return {"contact_count": 0}
        return {
            "contact_count": len(contacts),
            "contact_entity_a": [c[0] for c in contacts],
            "contact_label_a": [c[1] for c in contacts],
            "contact_entity_b": [c[2] for c in contacts],
            "contact_label_b": [c[3] for c in contacts],
            "contact_gap_um": [c[4] for c in contacts],
        }

    def get_aggregation(self, name: str):
        if name not in _COLUMNS:
            return None

        def agg(rows: List[Dict], _g_dims: Dict[str, Any]) -> Optional[Any]:
            # One chunk per object (run_chunk refuses fragments), so nothing to merge.
            return rows[0].get(name) if len(rows) == 1 else None

        return agg
