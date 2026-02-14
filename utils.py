"""
utils.py — Shared Utilities

Common helper classes and functions used across the surveillance
system: logging, configuration loading, geometry helpers, and
frame pre-processing utilities.
"""

import logging
import os
from typing import List, Tuple, Dict, Any, Optional

import numpy as np


# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────

class Logger:
    """
    Lightweight wrapper around Python's logging module.
    Provides consistent formatting across all subsystems.
    """

    def __init__(self, name: str = "Surveillance", level: int = logging.INFO):
        """
        Args:
            name:  Logger name (appears in log output).
            level: Logging verbosity level.
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def debug(self, message: str) -> None:
        self._logger.debug(message)


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

class ConfigLoader:
    """
    Loads and validates YAML/JSON configuration files used by
    subsystem components.
    """

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """
        Load a configuration file and return its contents as a dict.

        Args:
            filepath: Path to the config file (YAML or JSON).

        Returns:
            Parsed configuration dictionary.
        """
        # TODO: Implement YAML / JSON parsing with schema validation
        return {}

    @staticmethod
    def validate(config: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Ensure all required keys exist in the configuration.

        Args:
            config:        The loaded configuration dictionary.
            required_keys: List of keys that must be present.

        Returns:
            True if all required keys are present.

        Raises:
            ValueError: If any required key is missing.
        """
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
        return True


# ──────────────────────────────────────────────
#  Geometry Helpers
# ──────────────────────────────────────────────

class GeometryUtils:
    """Static methods for common spatial / geometric calculations."""

    @staticmethod
    def compute_iou(box_a: List[int], box_b: List[int]) -> float:
        """
        Compute Intersection-over-Union between two bounding boxes.

        Args:
            box_a: [x1, y1, x2, y2]
            box_b: [x1, y1, x2, y2]

        Returns:
            IoU value between 0.0 and 1.0.
        """
        # TODO: Implement IoU calculation
        return 0.0

    @staticmethod
    def compute_centroid(bbox: List[int]) -> Tuple[int, int]:
        """
        Calculate the center point of a bounding box.

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def euclidean_distance(point_a: Tuple, point_b: Tuple) -> float:
        """
        Compute Euclidean distance between two 2-D points.

        Args:
            point_a: (x, y)
            point_b: (x, y)

        Returns:
            Distance as a float.
        """
        return float(np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2))

    @staticmethod
    def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """
        Test whether a point lies inside a polygon (zone boundary).

        Args:
            point:   (x, y) coordinate.
            polygon: Ordered list of (x, y) vertices.

        Returns:
            True if the point is inside the polygon.
        """
        # TODO: Implement ray-casting or winding-number algorithm
        return False


# ──────────────────────────────────────────────
#  Frame Utilities
# ──────────────────────────────────────────────

class FrameUtils:
    """Pre-processing and post-processing helpers for video frames."""

    @staticmethod
    def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Resize a frame to the specified dimensions.

        Args:
            frame:  Input image (NumPy array).
            width:  Target width in pixels.
            height: Target height in pixels.

        Returns:
            Resized frame.
        """
        # TODO: Use cv2.resize with appropriate interpolation
        return frame

    @staticmethod
    def draw_bounding_box(
        frame: np.ndarray,
        bbox: List[int],
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw a labeled bounding box on the frame.

        Args:
            frame:     Image to annotate.
            bbox:      [x1, y1, x2, y2]
            label:     Text label to display above the box.
            color:     BGR color tuple.
            thickness: Line thickness in pixels.

        Returns:
            Annotated frame.
        """
        # TODO: Use cv2.rectangle and cv2.putText
        return frame

    @staticmethod
    def save_snapshot(frame: np.ndarray, output_dir: str, filename: str) -> str:
        """
        Save a frame to disk as an image file.

        Args:
            frame:      Image to save.
            output_dir: Directory path for the output.
            filename:   Output filename (e.g., 'alert_001.jpg').

        Returns:
            Full path to the saved image.
        """
        # TODO: Ensure output_dir exists, write with cv2.imwrite
        path = os.path.join(output_dir, filename)
        return path
