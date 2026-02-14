"""
detector.py — Object / Person Detection Module

Wraps a deep-learning object detection model (e.g., YOLOv8, SSD)
to locate people and objects of interest in each video frame.
"""

from typing import List, Dict, Any
import numpy as np
from ultralytics import YOLO


class Detection:
    """
    Data class representing a single detection in a frame.

    Attributes:
        bbox:       Bounding box as [x1, y1, x2, y2].
        confidence: Detection confidence score (0.0 – 1.0).
        class_id:   Integer class identifier.
        class_name: Human-readable class label.
    """

    def __init__(
        self,
        bbox: List[int],
        confidence: float,
        class_id: int,
        class_name: str = "person",
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def __repr__(self) -> str:
        return (
            f"Detection(class={self.class_name}, "
            f"conf={self.confidence:.2f}, bbox={self.bbox})"
        )


class ObjectDetector:
    """
    Encapsulates model loading and inference for object detection.
    Designed to be model-agnostic — swap the backend by changing
    the _load_model / _infer implementations.
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Args:
            model_path:           Path to the pre-trained model weights.
            confidence_threshold: Minimum confidence to keep a detection.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """
        Load the detection model from disk.

        Returns:
            The loaded YOLO model instance.
        """
        # Load YOLOv8 model — fall back to the nano variant if no path is given
        model_file = self.model_path if self.model_path else "yolov8n.pt"
        return YOLO(model_file)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame and return detections.

        Args:
            frame: BGR image as a NumPy array (H × W × 3).

        Returns:
            List of Detection objects above the confidence threshold.
        """
        # TODO: Pre-process frame, run model inference, post-process results
        raw_results = self._infer(frame)
        detections = self._postprocess(raw_results)
        return detections

    def _infer(self, frame: np.ndarray) -> Any:
        """
        Execute the raw model forward pass.

        Args:
            frame: Pre-processed input tensor / array.

        Returns:
            Raw YOLOv8 Results object.
        """
        # Run YOLOv8 inference on the frame
        results = self.model(frame)
        return results

    def _postprocess(self, raw_results: Any) -> List[Detection]:
        """
        Convert raw model outputs into a list of Detection objects.
        Apply confidence filtering and non-maximum suppression.

        Args:
            raw_results: Raw output from _infer().

        Returns:
            Filtered list of Detection instances.
        """
        detections: List[Detection] = []

        # Extract boxes from the first (and typically only) result
        boxes = raw_results[0].boxes
        class_names = raw_results[0].names  # {id: name} mapping from the model

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names.get(class_id, "unknown")
            confidence = float(box.conf[0])

            # Keep only "person" detections above the confidence threshold
            if class_name != "person":
                continue
            if confidence < self.confidence_threshold:
                continue

            # Convert bounding-box coordinates to integers
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return detections

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the minimum confidence threshold at runtime."""
        self.confidence_threshold = threshold
