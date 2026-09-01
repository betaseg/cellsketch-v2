from pathlib import Path


def get_viewer_extension_dir() -> Path:
    """Directory holding extension.json and this extension's viewer plugins."""
    return Path(__file__).parent / "viewer"
