"""Entry points that make the report self-describing.

These are not how a run happens: `process` builds its own pipeline, because an object cannot
be split (see the README). They are how PixelPatrol's schema catalogue finds this package's
``OUTPUT_SCHEMA`` and ``DESCRIPTIONS``, which is what writes a description into every column's
field metadata when the parquet is saved. Without them the report still loads and every widget
still works, but nothing in it says what a column means.
"""

from pixel_patrol_anatomy.plugins.loaders.object_loader import ObjectLoader
from pixel_patrol_anatomy.plugins.processors.contacts import ContactsProcessor
from pixel_patrol_anatomy.plugins.processors.instances import InstanceProcessor
from pixel_patrol_anatomy.plugins.processors.mesh import MeshProcessor
from pixel_patrol_anatomy.plugins.processors.morphology import MorphologyProcessor


def register_loader_plugins():
    return [ObjectLoader]


def register_processor_plugins():
    return [MorphologyProcessor, InstanceProcessor, ContactsProcessor, MeshProcessor]
