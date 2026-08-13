"""Entry points PixelPatrol calls to discover this extension's plugins."""

from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CellLoader
from pixel_patrol_cellsketch.plugins.processors.morphology import MorphologyProcessor


def register_loader_plugins():
    return [CellLoader]


def register_processor_plugins():
    return [MorphologyProcessor]
