"""
behavior.py — Behavioral Analysis Module

Interprets tracked movement patterns to classify human behaviors
such as loitering (lingering in a small area) and fast movement
(running / sprinting). Uses only centroid history — no external
libraries required.
"""

import math
from typing import List, Dict, Tuple


class BehaviorAnalyzer:
    """
    Analyzes per-track position history to detect behavioral
    patterns like loitering and fast movement.

    Attributes:
        history:          Per-track state keyed by track ID.
        loiter_radius:    Max displacement (px) to count as stationary.
        loiter_frames:    Min frames within the radius to flag loitering.
        speed_threshold:  Pixels-per-frame above which motion is "fast".
    """

    def __init__(
        self,
        loiter_radius: float = 30.0,
        loiter_frames: int = 50,
        speed_threshold: float = 20.0,
    ):
        """
        Args:
            loiter_radius:   Maximum displacement from the starting
                             position to still be considered stationary.
            loiter_frames:   Number of frames within the radius before
                             loitering is flagged.
            speed_threshold: Average speed (px/frame) above which
                             motion is classified as fast.
        """
        self.history: Dict[int, dict] = {}
        self.loiter_radius = loiter_radius
        self.loiter_frames = loiter_frames
        self.speed_threshold = speed_threshold

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance between two 2-D points."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _total_distance(self, positions: List[Tuple[int, int]]) -> float:
        """
        Sum of segment distances along the full trajectory.

        Args:
            positions: Ordered list of (x, y) centroids.

        Returns:
            Cumulative distance traveled (pixels).
        """
        dist = 0.0
        for i in range(1, len(positions)):
            dist += self._euclidean(positions[i - 1], positions[i])
        return dist

    # ──────────────────────────────────────────
    #  Behavior Detection
    # ──────────────────────────────────────────

    def _is_loitering(self, positions: List[Tuple[int, int]]) -> bool:
        """
        Detect loitering: object stays within a small radius
        of its starting position for more than loiter_frames.

        Args:
            positions: Ordered centroid history for one track.

        Returns:
            True if loitering is detected.
        """
        if len(positions) < self.loiter_frames:
            return False

        # Check the most recent loiter_frames positions
        window = positions[-self.loiter_frames:]
        anchor = window[0]

        for pos in window[1:]:
            if self._euclidean(anchor, pos) > self.loiter_radius:
                return False

        return True

    def _is_fast_movement(self, speed: float) -> bool:
        """
        Detect fast movement: average speed exceeds threshold.

        Args:
            speed: Average speed in pixels per frame.

        Returns:
            True if fast movement is detected.
        """
        return speed > self.speed_threshold

    # ──────────────────────────────────────────
    #  Core Analysis
    # ──────────────────────────────────────────

    def analyze(self, tracks: List[dict], frame_index: int) -> List[dict]:
        """
        Update position history and classify behaviors for every
        active track.

        Args:
            tracks:      List of tracked-object dicts from ObjectTracker:
                         [{"id": int, "bbox": [...], "centroid": (cx, cy)}, ...]
            frame_index: Current frame number (used for timing).

        Returns:
            List of behavior reports:
            [
                {
                    "id":            int,
                    "loitering":     bool,
                    "fast_movement": bool,
                    "speed":         float   # average px/frame
                },
                ...
            ]
        """
        results: List[dict] = []

        # Set of IDs seen this frame (for stale-entry cleanup)
        active_ids = set()

        for track in tracks:
            track_id = track["id"]
            centroid = track["centroid"]
            active_ids.add(track_id)

            # ── Update history ────────────────────────
            if track_id not in self.history:
                self.history[track_id] = {
                    "positions": [],
                    "first_seen": frame_index,
                    "last_seen": frame_index,
                }

            self.history[track_id]["positions"].append(centroid)
            self.history[track_id]["last_seen"] = frame_index

            # ── Compute metrics ───────────────────────
            positions = self.history[track_id]["positions"]
            total_dist = self._total_distance(positions)

            elapsed = len(positions)
            speed = total_dist / elapsed if elapsed > 0 else 0.0

            # ── Classify behaviors ────────────────────
            loitering = self._is_loitering(positions)
            fast_movement = self._is_fast_movement(speed)

            results.append({
                "id": track_id,
                "loitering": loitering,
                "fast_movement": fast_movement,
                "speed": round(speed, 2),
            })

        # ── Cleanup stale entries no longer tracked ───
        stale_ids = [tid for tid in self.history if tid not in active_ids]
        for tid in stale_ids:
            del self.history[tid]

        return results
