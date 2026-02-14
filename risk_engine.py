"""
risk_engine.py — Risk Assessment Engine

Accumulates a cumulative risk score per tracked person based on
detected behaviors (loitering, fast movement, zone breach). Triggers an alert
when the score exceeds a configurable threshold. No external
libraries required.
"""

from typing import List, Dict


class RiskEngine:
    """
    Scores and classifies risk based on behavioral analysis results.
    Risk points are accumulated over time — once a track's cumulative
    score meets or exceeds the threshold, an alert is raised.

    Attributes:
        risk_scores:      Cumulative score per track ID.
        threshold:        Score at or above which an alert is triggered.
        loiter_points:    Points added per frame of loitering.
        fast_move_points: Points added per frame of fast movement.
        zone_points:      Points added per frame of restricted zone breach.
    """

    def __init__(
        self,
        threshold: int = 30,
        loiter_points: int = 10,
        fast_move_points: int = 15,
        zone_points: int = 20,
    ):
        """
        Args:
            threshold:        Cumulative score that triggers an alert.
            loiter_points:    Risk points awarded when loitering is detected.
            fast_move_points: Risk points awarded when fast movement is detected.
            zone_points:      Risk points awarded when a restricted zone breach
                              is detected.
        """
        self.risk_scores: Dict[int, int] = {}
        self.threshold = threshold
        self.loiter_points = loiter_points
        self.fast_move_points = fast_move_points
        self.zone_points = zone_points

    # ──────────────────────────────────────────
    #  Core Evaluation
    # ──────────────────────────────────────────

    def evaluate(self, behaviors: List[dict]) -> List[dict]:
        """
        Update cumulative risk scores and determine alert status
        for every tracked person.

        Args:
            behaviors: List of behavior dicts from BehaviorAnalyzer:
                       [
                           {
                               "id":            int,
                               "loitering":     bool,
                               "fast_movement": bool,
                               "zone_breach":   bool,
                               "speed":         float
                           },
                           ...
                       ]

        Returns:
            List of risk reports:
            [
                {
                    "id":         int,
                    "risk_score": int,
                    "alert":      bool
                },
                ...
            ]
        """
        reports: List[dict] = []

        for entry in behaviors:
            track_id = entry["id"]

            # Initialize score for newly seen IDs
            if track_id not in self.risk_scores:
                self.risk_scores[track_id] = 0

            # Accumulate risk points based on detected behaviors
            if entry["loitering"]:
                self.risk_scores[track_id] += self.loiter_points

            if entry["fast_movement"]:
                self.risk_scores[track_id] += self.fast_move_points

            if entry.get("zone_breach", False):
                self.risk_scores[track_id] += self.zone_points

            # Determine alert status
            score = self.risk_scores[track_id]
            alert = score >= self.threshold

            reports.append({
                "id": track_id,
                "risk_score": score,
                "alert": alert,
            })

        return reports

    # ──────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────

    def get_score(self, track_id: int) -> int:
        """
        Retrieve the current cumulative score for a single track.

        Args:
            track_id: The tracked person's unique ID.

        Returns:
            Cumulative risk score (0 if unknown ID).
        """
        return self.risk_scores.get(track_id, 0)

    def reset(self, track_id: int = None) -> None:
        """
        Reset risk scores.

        Args:
            track_id: If provided, reset only this track.
                      If None, reset all scores.
        """
        if track_id is not None:
            self.risk_scores.pop(track_id, None)
        else:
            self.risk_scores.clear()
