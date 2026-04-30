"""Visualization module for KeySG scene graphs.

Usage:
    from KeySG.visualization import KeySGVisualizer

    KeySGVisualizer("output/pipeline/ScanNet/scene0011_00").run()

Or from command line:
    keysg-vis --scene_dir output/pipeline/ScanNet/scene0011_00
    python -m KeySG.visualization.visualizer --scene_dir <path>
"""

from .visualizer import KeySGVisualizer

__all__ = ["KeySGVisualizer"]
