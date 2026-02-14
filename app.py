"""
app.py — Application Entry Point

Main orchestrator for the AI-powered Behavioral Surveillance
& Risk Assessment System. Processes a video file through the
full pipeline (detection → tracking → behavior → risk) and
writes an annotated output video. Designed for batch processing
and Streamlit integration.
"""

import cv2
import numpy as np
from detector import ObjectDetector
from tracker import ObjectTracker
from behavior import BehaviorAnalyzer
from risk_engine import RiskEngine


class SurveillanceApp:
    """
    Core application controller that coordinates detection, tracking,
    behavior analysis, and risk assessment across video frames.
    Outputs an annotated video and returns summary metrics.
    """

    def __init__(self):
        """
        Initialize all subsystem components and summary accumulators.
        """
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.risk_engine = RiskEngine()
        self.frame_index = 0

        # Restricted zone boundaries — list of (x1, y1, x2, y2)
        self.restricted_zones: list = []

        # ── Summary accumulators ──────────────────
        self.all_track_ids: set = set()
        self.total_alerts: int = 0
        self.max_risk: int = 0
        self.total_zone_breaches: int = 0

        # ── Zone duration tracking ────────────────
        self.zone_entry_time: dict = {}   # track_id -> frame_index when entered
        self.active_alarm: bool = False

        # ── Event logging ─────────────────────────
        self.event_log: list = []
        self._logged_zone_entries: set = set()  # avoid duplicate entry events

        print("[SurveillanceApp] All subsystems initialized successfully.")

    # ──────────────────────────────────────────────
    #  Reset Accumulators
    # ──────────────────────────────────────────────

    def _reset_state(self):
        """
        Reset all internal state for a fresh processing run.
        Called at the start of process_video() and can also be
        called externally before streaming frames.
        """
        self.frame_index = 0
        self.tracker = ObjectTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.risk_engine = RiskEngine()
        self.all_track_ids = set()
        self.total_alerts = 0
        self.max_risk = 0
        self.total_zone_breaches = 0
        self.zone_entry_time = {}
        self.active_alarm = False
        self.event_log = []
        self._logged_zone_entries = set()

    # ──────────────────────────────────────────────
    #  Zone Configuration
    # ──────────────────────────────────────────────

    def set_restricted_zones(self, zones: list):
        """
        Define one or more restricted zones for breach detection.

        Args:
            zones: List of (x1, y1, x2, y2) tuples, each defining
                   a rectangular restricted area.
        """
        self.restricted_zones = zones

    # ──────────────────────────────────────────────
    #  Per-Frame Processing (used by both batch & streaming)
    # ──────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Run the full surveillance pipeline on a single frame and
        return the annotated frame along with cumulative metrics.

        This method is designed for real-time / streaming use
        (e.g., Streamlit frame-by-frame rendering).

        Args:
            frame: BGR image as a NumPy array (H × W × 3).

        Returns:
            (annotated_frame, summary_dict)
            where summary_dict is:
            {
                "total_tracks":   int,  # unique track IDs seen so far
                "total_alerts":   int,  # total alert-flagged frames so far
                "max_risk":       int,  # highest cumulative risk score
                "zone_breaches":  int   # total zone-breach detections
            }
        """
        self.frame_index += 1

        # ── Full Pipeline ─────────────────────────
        # Step 1 — Detect persons
        detections = self.detector.detect(frame)

        # Step 2 — Update tracker with detections
        tracks = self.tracker.update(detections)

        # Step 3 — Analyze behaviors from tracked trajectories
        behaviors = self.behavior_analyzer.analyze(tracks, self.frame_index)

        # Step 3.5 — Check restricted zone breach per track
        zone_breach_ids = set()
        for track in tracks:
            cx, cy = track["centroid"]
            for zone in self.restricted_zones:
                zx1, zy1, zx2, zy2 = zone
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    zone_breach_ids.add(track["id"])
                    break  # one breach is enough per track

        # Inject zone_breach flag into each behavior dict
        for beh in behaviors:
            beh["zone_breach"] = beh["id"] in zone_breach_ids

        # ── Zone duration tracking ────────────────
        # Record entry time for tracks that just entered a zone;
        # remove tracks that have left all zones.
        current_track_ids = {t["id"] for t in tracks}
        for tid in current_track_ids:
            if tid in zone_breach_ids:
                # Inside a zone — record entry time if new
                if tid not in self.zone_entry_time:
                    self.zone_entry_time[tid] = self.frame_index
                    # Log zone-entry event (once per entry)
                    if tid not in self._logged_zone_entries:
                        self._logged_zone_entries.add(tid)
                        self.event_log.append({
                            "frame": self.frame_index,
                            "event": f"Track {tid} entered restricted zone",
                        })
            else:
                # Outside all zones — clear entry time
                if tid in self.zone_entry_time:
                    self._logged_zone_entries.discard(tid)
                self.zone_entry_time.pop(tid, None)

        # Clean up stale entries for tracks that no longer exist
        stale = [tid for tid in self.zone_entry_time if tid not in current_track_ids]
        for tid in stale:
            self._logged_zone_entries.discard(tid)
            del self.zone_entry_time[tid]

        # Check for persistent breach (>= 10 seconds at 30 FPS)
        FPS = 30
        PERSISTENT_THRESHOLD_SEC = 10
        persistent_breach = False
        for tid, entry_frame in self.zone_entry_time.items():
            duration_sec = (self.frame_index - entry_frame) / FPS
            if duration_sec >= PERSISTENT_THRESHOLD_SEC:
                persistent_breach = True
                break

        # Log persistent breach event (once when it first fires)
        if persistent_breach and not self.active_alarm:
            self.event_log.append({
                "frame": self.frame_index,
                "event": "⚠️ Persistent breach detected (≥10 s)",
            })

        self.active_alarm = persistent_breach

        # Step 4 — Evaluate risk from observed behaviors
        risk_reports = self.risk_engine.evaluate(behaviors)

        # ── Accumulate summary metrics ────────────
        for track in tracks:
            self.all_track_ids.add(track["id"])

        for r in risk_reports:
            if r["alert"]:
                self.total_alerts += 1
                # Log alert event
                self.event_log.append({
                    "frame": self.frame_index,
                    "event": f"🚨 Alert triggered for Track {r['id']} (score={r['risk_score']})",
                })
            if r["risk_score"] > self.max_risk:
                self.max_risk = r["risk_score"]

        self.total_zone_breaches += len(zone_breach_ids)

        # ── Per-track analytics ────────────────────
        risk_map = {r["id"]: r for r in risk_reports}
        track_analytics = []
        for track in tracks:
            tid = track["id"]
            beh = {b["id"]: b for b in behaviors}.get(tid, {})
            rsk = risk_map.get(tid, {})
            score = rsk.get("risk_score", 0)

            # Risk level classification
            if score < 20:
                risk_level = "LOW"
            elif score < 40:
                risk_level = "MEDIUM"
            elif score < 70:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"

            # Zone duration
            zone_dur = 0.0
            if tid in self.zone_entry_time:
                zone_dur = round((self.frame_index - self.zone_entry_time[tid]) / 30, 1)

            track_analytics.append({
                "id": tid,
                "speed": beh.get("speed", 0.0),
                "risk_score": score,
                "risk_level": risk_level,
                "alert": rsk.get("alert", False),
                "zone_breach": tid in zone_breach_ids,
                "zone_duration": zone_dur,
            })

        # ── Build lookup maps for overlay ─────────
        behavior_map = {b["id"]: b for b in behaviors}
        # risk_map already built above

        # ── Draw all restricted zone rectangles (blue) ─
        for zone in self.restricted_zones:
            zx1, zy1, zx2, zy2 = zone
            cv2.rectangle(
                frame,
                (zx1, zy1), (zx2, zy2),
                (255, 0, 0), 2,
            )
            cv2.putText(
                frame, "RESTRICTED ZONE",
                (zx1, zy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 0), 2, cv2.LINE_AA,
            )

        # ── Draw overlay on frame ─────────────────
        for track in tracks:
            tid = track["id"]
            x1, y1, x2, y2 = track["bbox"]

            # Fetch behavior & risk data for this track
            beh = behavior_map.get(tid, {})
            risk = risk_map.get(tid, {})

            speed = beh.get("speed", 0.0)
            risk_score = risk.get("risk_score", 0)
            alert = risk.get("alert", False)
            zone_breach = beh.get("zone_breach", False)

            # Choose color — red if alert, green otherwise
            color = (0, 0, 255) if alert else (0, 255, 0)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Build label lines
            label_id = f"ID:{tid}"
            label_speed = f"Spd:{speed:.1f}"
            label_risk = f"Risk:{risk_score}"

            # Draw label background + text (stacked above top-left)
            labels = [label_id, label_speed, label_risk]
            if zone_breach:
                labels.append("ZONE BREACH")
                # Show dwell time if this track has an entry time
                if tid in self.zone_entry_time:
                    dwell = (self.frame_index - self.zone_entry_time[tid]) / 30
                    labels.append(f"DWELL:{dwell:.1f}s")
                    if dwell >= 10:
                        labels.append("PERSISTENT")
            if alert:
                labels.append("ALERT")

            y_offset = y1 - 4
            for label in reversed(labels):
                (lw, lh), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                # Background rectangle
                cv2.rectangle(
                    frame,
                    (x1, y_offset - lh - baseline),
                    (x1 + lw, y_offset + 2),
                    color,
                    cv2.FILLED,
                )
                # Text
                cv2.putText(
                    frame,
                    label,
                    (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y_offset -= lh + baseline + 4

        # ── Return annotated frame + cumulative summary
        summary = {
            "total_tracks": len(self.all_track_ids),
            "total_alerts": self.total_alerts,
            "max_risk": self.max_risk,
            "zone_breaches": self.total_zone_breaches,
            "persistent_breach": self.active_alarm,
            "zone_entry_ids": list(self.zone_entry_time.keys()),
            "track_analytics": track_analytics,
            "event_log": self.event_log[-50:],  # last 50 events
        }

        return frame, summary

    # ──────────────────────────────────────────────
    #  Batch Video Processing
    # ──────────────────────────────────────────────

    def process_video(self, input_path: str, output_path: str) -> dict:
        """
        Run the full surveillance pipeline on a video file and write
        the annotated result to disk.

        Args:
            input_path:  Path to the source video file.
            output_path: Path where the annotated output video is saved.

        Returns:
            Summary metrics dictionary:
            {
                "total_tracks":   int,  # unique track IDs seen
                "total_alerts":   int,  # total alert-flagged frames
                "max_risk":       int,  # highest cumulative risk score
                "zone_breaches":  int   # total zone-breach detections
            }
        """
        print(f"[SurveillanceApp] Processing: {input_path}")

        # 🔥 Reset state for fresh processing
        self._reset_state()

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {input_path}")
            return {"total_tracks": 0, "total_alerts": 0,
                    "max_risk": 0, "zone_breaches": 0}

        # Read video properties for VideoWriter
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # ── Frame loop ────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Delegate all per-frame logic to process_frame()
            annotated_frame, summary = self.process_frame(frame)

            # ── Write annotated frame to output ───
            writer.write(annotated_frame)

        # ── Cleanup ───────────────────────────────
        cap.release()
        writer.release()

        # Final summary (same as last process_frame return)
        summary = {
            "total_tracks": len(self.all_track_ids),
            "total_alerts": self.total_alerts,
            "max_risk": self.max_risk,
            "zone_breaches": self.total_zone_breaches,
            "persistent_breach": self.active_alarm,
            "zone_entry_ids": list(self.zone_entry_time.keys()),
            "track_analytics": [],
            "event_log": self.event_log[-50:],
        }

        print(f"[SurveillanceApp] Processing complete. Summary: {summary}")
        return summary


# ──────────────────────────────────────────────
#  Entry Point (for standalone testing)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = SurveillanceApp()
    result = app.process_video("sample_video.mp4", "output_video.mp4")
    print(result)
