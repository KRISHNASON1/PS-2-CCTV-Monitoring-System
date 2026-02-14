"""
tracker.py — Centroid-Based Multi-Object Tracker

Maintains identity persistence across frames by matching new
detections to existing tracked objects using Euclidean distance
between centroids. No external tracking libraries required.
"""

import math
from typing import List, Dict, Tuple


class ObjectTracker:
    """
    Simple centroid-based tracker that assigns and maintains unique
    IDs for detected persons across consecutive video frames.

    Attributes:
        next_id:     Auto-incrementing counter for new track IDs.
        objects:     Active tracks — maps track ID → centroid (cx, cy).
        bboxes:      Active bounding boxes — maps track ID → [x1, y1, x2, y2].
        lost_frames: Maps track ID → number of consecutive unmatched frames.
        max_lost:    Frames after which an unmatched track is removed.
        max_dist:    Maximum pixel distance to consider a match (default 50).
    """

    def __init__(self, max_lost: int = 30, max_dist: float = 50.0):
        """
        Args:
            max_lost: Maximum consecutive frames a track can go
                      unmatched before it is removed.
            max_dist: Maximum Euclidean distance (pixels) to accept
                      a detection–track match.
        """
        self.next_id: int = 0
        self.objects: Dict[int, Tuple[int, int]] = {}      # id → centroid
        self.bboxes: Dict[int, List[int]] = {}              # id → bbox
        self.lost_frames: Dict[int, int] = {}               # id → lost count
        self.max_lost = max_lost
        self.max_dist = max_dist

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _centroid(bbox: List[int]) -> Tuple[int, int]:
        """
        Compute the center point of a bounding box.

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            (cx, cy) center coordinates.
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Euclidean distance between two 2-D points.

        Args:
            a: (x, y)
            b: (x, y)

        Returns:
            Distance as a float.
        """
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # ──────────────────────────────────────────
    #  Core Update
    # ──────────────────────────────────────────

    def update(self, detections: list) -> List[dict]:
        """
        Associate incoming detections with existing tracks using
        greedy nearest-centroid matching.

        Steps:
            1. Compute centroids for all new detections.
            2. For each detection, find the closest existing track.
               • If distance < max_dist → re-use that track ID.
               • Otherwise → register a new track ID.
            3. Mark unmatched tracks as lost; remove them if they
               exceed max_lost consecutive frames.

        Args:
            detections: List of Detection objects from the detector.

        Returns:
            List of tracked-object dicts:
            [
                {
                    "id":       int,
                    "bbox":     [x1, y1, x2, y2],
                    "centroid": (cx, cy)
                },
                ...
            ]
        """
        new_centroids: List[Tuple[int, int]] = []
        new_bboxes: List[List[int]] = []

        # Step 1 — Compute centroids for every incoming detection
        for det in detections:
            centroid = self._centroid(det.bbox)
            new_centroids.append(centroid)
            new_bboxes.append(det.bbox)

        # Prepare structures for this frame
        updated_objects: Dict[int, Tuple[int, int]] = {}
        updated_bboxes: Dict[int, List[int]] = {}
        matched_ids: set = set()         # existing IDs that got matched
        matched_det_indices: set = set() # detection indices that got matched

        # Step 2 — Greedy matching: for each existing track find nearest detection
        for obj_id, obj_centroid in self.objects.items():
            best_dist = float("inf")
            best_idx = -1

            for idx, det_centroid in enumerate(new_centroids):
                if idx in matched_det_indices:
                    continue  # already assigned to another track
                dist = self._euclidean(obj_centroid, det_centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_dist < self.max_dist and best_idx != -1:
                # Match found — keep the same ID
                updated_objects[obj_id] = new_centroids[best_idx]
                updated_bboxes[obj_id] = new_bboxes[best_idx]
                self.lost_frames[obj_id] = 0
                matched_ids.add(obj_id)
                matched_det_indices.add(best_idx)

        # Step 3a — Register new tracks for unmatched detections
        for idx in range(len(new_centroids)):
            if idx not in matched_det_indices:
                updated_objects[self.next_id] = new_centroids[idx]
                updated_bboxes[self.next_id] = new_bboxes[idx]
                self.lost_frames[self.next_id] = 0
                self.next_id += 1

        # Step 3b — Handle unmatched existing tracks (increment lost counter)
        for obj_id in list(self.objects.keys()):
            if obj_id not in matched_ids:
                self.lost_frames[obj_id] = self.lost_frames.get(obj_id, 0) + 1
                # Keep the track alive until it exceeds max_lost
                if self.lost_frames[obj_id] <= self.max_lost:
                    updated_objects[obj_id] = self.objects[obj_id]
                    updated_bboxes[obj_id] = self.bboxes.get(obj_id, [0, 0, 0, 0])
                else:
                    # Remove stale track
                    self.lost_frames.pop(obj_id, None)

        # Commit the updated state
        self.objects = updated_objects
        self.bboxes = updated_bboxes

        # Step 4 — Build the output list
        tracked: List[dict] = []
        for obj_id, centroid in self.objects.items():
            tracked.append({
                "id": obj_id,
                "bbox": self.bboxes[obj_id],
                "centroid": centroid,
            })

        return tracked
