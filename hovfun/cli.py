"""CLI entry points for KeySG.

Installed as:
    keysg-build   →  build the KeySG scene graph from an RGB-D dataset
    keysg-vis     →  open the interactive Viser visualizer for a processed scene
"""

from __future__ import annotations


def build() -> None:
    """Entry point for `keysg-build`. Delegates to the Hydra-based pipeline."""
    import sys
    import os

    # Ensure the repo root is importable when running via entry point
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    import main_pipeline  # noqa: F401 — triggers hydra @main decorator

    main_pipeline.main()  # type: ignore[attr-defined]


def visualize() -> None:
    """Entry point for `keysg-vis`. Opens the Viser scene visualizer."""
    from hovfun.visualization.visualizer import main

    main()
