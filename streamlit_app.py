"""
streamlit_app.py — AI Behavioral Surveillance Dashboard
Stable while-loop streaming architecture (no st.rerun in stream loop)
"""

import time
import os
import platform
import subprocess
from datetime import datetime
import streamlit as st
import cv2
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from app import SurveillanceApp


# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Behavioral Surveillance",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════
#  CSS STYLING
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #fff !important;
    color: #111 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}
.block-container {
    max-width: 100% !important;
    padding: 60px 40px !important;
}
.stButton > button {
    width: 100%;
    min-height: 46px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    background: #111 !important;
    color: #fff !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    transition: all 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
}
[data-testid="stMetric"] {
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 14px;
    padding: 16px 18px;
}
[data-testid="stMetricLabel"] {
    color: #777 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #111 !important;
    font-weight: 800 !important;
    font-size: 1.3rem !important;
}
.m-card {
    background: #fff;
    border: 1.5px solid #eaeaea;
    border-radius: 20px;
    padding: 2.6rem 2rem 2.2rem;
    text-align: center;
    transition: all 0.25s;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}
.m-card:hover {
    border-color: #d4d4d4;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    transform: translateY(-4px);
}
.m-card-icon { font-size: 2.8rem; margin-bottom: 1rem; }
.m-card-title { font-size: 1.15rem; font-weight: 800; color: #111; margin-bottom: 0.4rem; }
.m-card-desc { font-size: 0.82rem; color: #999; line-height: 1.55; }
.ctrl-panel {
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 20px;
    padding: 1.5rem 1.3rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    max-height: 86vh;
    overflow-y: auto;
}
.vid-card {
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 20px;
    padding: 5px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}
.vid-card img { border-radius: 16px; width: 100%; display: block; }
.sec-label {
    font-size: 0.68rem; font-weight: 700; color: #999;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 0.7rem 0 0.5rem 0;
}
.b-red { display:inline-block; background:#ef4444; color:#fff; font-size:0.7rem; font-weight:700; padding:4px 12px; border-radius:6px; }
.b-amber { display:inline-block; background:#f59e0b; color:#fff; font-size:0.7rem; font-weight:700; padding:4px 12px; border-radius:6px; }
.b-green { display:inline-block; background:#22c55e; color:#fff; font-size:0.7rem; font-weight:700; padding:4px 12px; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════

defaults = {
    "page": "landing",
    "run": False,
    "cap": None,
    "app_instance": None,
    "zones": [],
    "zone_frame": None,
    "risk_history": [],
    "event_log": [],
    "track_table": [],
    "track_table_data": [],
    "risk_threshold": 30,
    "speed_threshold": 20,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def log_event(event: str, person_id: int = -1, risk_score: int = 0):
    st.session_state["event_log"].append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "person_id": person_id,
        "event": event,
        "risk_score": risk_score,
    })


def cleanup():
    if st.session_state.get("cap") is not None:
        st.session_state["cap"].release()
    st.session_state.update({
        "cap": None, "run": False, "app_instance": None,
    })


def render_zone_canvas(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    h, w = frame.shape[:2]
    scale = min(680 / w, 1.0)
    cw, ch = int(w * scale), int(h * scale)

    st.markdown('<p class="sec-label">🎯 Draw Restricted Zones</p>', unsafe_allow_html=True)
    result = st_canvas(
        fill_color="rgba(239,68,68,0.12)", stroke_width=2, stroke_color="#ef4444",
        background_image=pil, update_streamlit=True,
        height=ch, width=cw, drawing_mode="rect", key="zone_canvas",
    )

    if result.json_data is not None:
        objs = result.json_data.get("objects", [])
        zones = []
        for o in objs:
            if o.get("type") == "rect":
                x1 = int(o["left"] / scale)
                y1 = int(o["top"] / scale)
                x2 = int((o["left"] + o["width"]) / scale)
                y2 = int((o["top"] + o["height"]) / scale)
                if x2 > x1 and y2 > y1:
                    zones.append((x1, y1, x2, y2))
        st.session_state["zones"] = zones

    n = len(st.session_state["zones"])
    if n:
        st.markdown(f'<span class="b-amber">{n} zone(s) defined</span>', unsafe_allow_html=True)
    else:
        st.caption("Draw rectangles to define zones.")


# ═══════════════════════════════════════════════════════════
#  LANDING PAGE
# ═══════════════════════════════════════════════════════════

def render_landing():
    st.markdown("""
    <style>
    .hero { display:flex; flex-direction:column; justify-content:center; align-items:center; height:100vh; text-align:center; }
    .hero-title { font-size:96px; font-weight:900; line-height:1.05; margin-bottom:32px; color:#000; }
    .hero-subtitle { font-size:22px; color:#555; max-width:700px; margin-bottom:60px; line-height:1.5; }
    .brand { position:fixed; top:32px; left:60px; font-size:22px; font-weight:700; letter-spacing:3px; color:#111; }
    </style>
    <div class="brand">Enclope</div>
    <div class="hero">
        <div class="hero-title">AI Behavioral<br>Surveillance Platform</div>
        <div class="hero-subtitle">Real-time intelligent monitoring with behavioral analysis and automated risk detection.</div>
    </div>
    """, unsafe_allow_html=True)

    _, c, _ = st.columns([1, 1, 1])
    with c:
        if st.button("Start Monitoring →", use_container_width=True):
            st.session_state["page"] = "mode"
            st.rerun()


# ═══════════════════════════════════════════════════════════
#  MODE SELECTION PAGE
# ═══════════════════════════════════════════════════════════

def render_mode():
    st.markdown("")
    _, c, _ = st.columns([1, 3, 1])
    with c:
        st.markdown('<p style="text-align:center;font-size:1.5rem;font-weight:800;margin:0 0 0.15rem 0;">Choose Your Mode</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;color:#999;font-size:0.88rem;margin:0 0 2rem 0;">Select how you want to provide video input</p>', unsafe_allow_html=True)

    _, cl, _, cr, _ = st.columns([1.2, 3, 0.3, 3, 1.2])

    with cl:
        st.markdown('<div class="m-card"><div class="m-card-icon">📁</div><div class="m-card-title">Upload Footage</div><div class="m-card-desc">Analyze a pre-recorded MP4 video with full pipeline processing.</div></div>', unsafe_allow_html=True)
        if st.button("Select Upload", use_container_width=True):
            st.session_state["page"] = "upload"
            st.rerun()

    with cr:
        st.markdown('<div class="m-card"><div class="m-card-icon">📷</div><div class="m-card-title">Live Camera</div><div class="m-card-desc">Monitor in real-time through webcam with instant detection and alerts.</div></div>', unsafe_allow_html=True)
        if st.button("Select Camera", use_container_width=True):
            st.session_state["page"] = "webcam"
            st.rerun()


# ═══════════════════════════════════════════════════════════
#  WEBCAM MONITORING PAGE
# ═══════════════════════════════════════════════════════════

def render_webcam():
    # Safety check
    if "zones" not in st.session_state:
        st.session_state["zones"] = []
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### 📷 Live Camera Monitoring")
        st.markdown("<span style='color:#666;'>Capture a frame, draw zones, then start monitoring</span>", unsafe_allow_html=True)
    with col2:
        if st.button("← Back"):
            cleanup()
            st.session_state["zone_frame"] = None
            st.session_state["page"] = "mode"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Zone setup
    if not st.session_state["run"]:
        if st.button("📸 Capture Frame for Zone Setup"):
            tc = cv2.VideoCapture(0)
            ok, snap = tc.read()
            tc.release()
            if ok:
                st.session_state["zone_frame"] = snap
            else:
                st.error("Could not access webcam.")

    if st.session_state["zone_frame"] is not None and not st.session_state["run"]:
        render_zone_canvas(st.session_state["zone_frame"])

    # Layout
    col_v, col_p = st.columns([3, 1.2], gap="large")

    with col_v:
        st.markdown('<div class="vid-card">', unsafe_allow_html=True)
        frame_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        status_ph = st.empty()
        
        # Real-time tracking table below video
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="sec-label">📊 Active Tracks</p>', unsafe_allow_html=True)
        tracks_table_ph = st.empty()

    with col_p:
        st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)
        
        st.markdown('<p class="sec-label">System Controls</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        start = c1.button("▶ Start", use_container_width=True)
        stop = c2.button("⏹ Stop", use_container_width=True)
        c3, c4 = st.columns(2)
        reset = c4.button("🔄 Reset", use_container_width=True)

        st.markdown('<p class="sec-label">Sensitivity</p>', unsafe_allow_html=True)
        st.session_state["risk_threshold"] = st.slider("Risk Threshold", 10, 100, st.session_state["risk_threshold"])
        st.session_state["speed_threshold"] = st.slider("Speed (px/frame)", 5, 50, st.session_state["speed_threshold"])

        st.markdown('<p class="sec-label">Live Analytics</p>', unsafe_allow_html=True)
        analytics_ph = st.empty()

        st.markdown('<p class="sec-label">Live Track Data</p>', unsafe_allow_html=True)

        if st.session_state["track_table_data"]:
            df = pd.DataFrame(st.session_state["track_table_data"])
            st.dataframe(df.tail(25), use_container_width=True, height=220)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Session CSV",
                data=csv,
                file_name="session_track_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("No tracking data yet.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Button handlers
    if stop:
        cleanup()
        log_event("System stopped by user")
        status_ph.info("Stopped.")
    if reset:
        log_event("🔄 System reset")
        cleanup()
        st.session_state["zones"] = []
        st.session_state["zone_frame"] = None
        st.rerun()
    if start and not st.session_state["run"]:
        st.session_state["cap"] = cv2.VideoCapture(0)
        st.session_state["app_instance"] = SurveillanceApp()
        st.session_state["app_instance"].set_restricted_zones(st.session_state["zones"])
        st.session_state["app_instance"]._reset_state()
        st.session_state["run"] = True
        st.session_state["risk_history"] = []
        log_event("▶ Monitoring started (webcam)")

    # ─────────────────────────────────────────────
    # SIMPLE STABLE STREAMING LOOP
    # ─────────────────────────────────────────────

    if st.session_state["run"] and st.session_state["cap"]:

        cap = st.session_state["cap"]
        app_inst = st.session_state["app_instance"]

        while st.session_state["run"]:

            ret, frame = cap.read()
            if not ret:
                cleanup()
                status_ph.error("❌ Failed to read from webcam.")
                break

            # Apply thresholds
            app_inst.risk_engine.threshold = st.session_state["risk_threshold"]
            app_inst.behavior_analyzer.speed_threshold = float(st.session_state["speed_threshold"])

            # Process frame
            annotated, summary = app_inst.process_frame(frame)

            # Collect track analytics
            track_analytics = summary.get("track_analytics", [])

            for t in track_analytics:
                record = {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Track ID": t.get("id"),
                    "Speed": round(t.get("speed", 0), 2),
                    "Risk Score": t.get("risk_score", 0),
                    "Risk Level": t.get("risk_level", "N/A"),
                    "Alert": t.get("alert", False),
                    "Zone Breach": t.get("zone_breach", False),
                }
                st.session_state["track_table_data"].append(record)

            # Limit storage to last 1000 records
            if len(st.session_state["track_table_data"]) > 1000:
                st.session_state["track_table_data"] = st.session_state["track_table_data"][-1000:]

            # Display frame
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_ph.image(rgb, channels="RGB", use_container_width=True)

            # Simple analytics
            with analytics_ph.container():
                st.metric("Total Alerts", app_inst.total_alerts)
                st.metric("Zone Breaches", app_inst.total_zone_breaches)
                st.metric("Highest Risk", app_inst.max_risk)

            # Update real-time tracking table (current frame only)
            with tracks_table_ph.container():
                if track_analytics:
                    current_tracks = []
                    current_time = datetime.now().strftime("%H:%M:%S")
                    for t in track_analytics:
                        zone_dur = t.get("zone_duration", 0)
                        current_tracks.append({
                            "Person ID": t.get("id"),
                            "Current Time": current_time,
                            "Speed": round(t.get("speed", 0), 2),
                            "Risk Score": t.get("risk_score", 0),
                            "Risk Level": t.get("risk_level", "N/A"),
                            "Alert": "🚨 YES" if t.get("alert") else "No",
                            "Zone Breach": "⚠️ YES" if t.get("zone_breach") else "No",
                            "Zone Duration (s)": f"{zone_dur:.1f}" if zone_dur > 0 else "-",
                        })
                    df_current = pd.DataFrame(current_tracks)
                    st.dataframe(df_current, use_container_width=True, height=400)
                else:
                    st.caption("No active tracks in current frame.")

            # Basic beep
            if summary.get("zone_entry_ids"):
                if platform.system() == "Darwin":
                    subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif platform.system() == "Windows":
                    try:
                        import winsound
                        winsound.Beep(1000, 120)
                    except ImportError:
                        pass

            time.sleep(0.03)

        cleanup()


# ═══════════════════════════════════════════════════════════
#  UPLOAD MONITORING PAGE
# ═══════════════════════════════════════════════════════════

def render_upload():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### 📁 Upload Footage")
        st.markdown("<span style='color:#666;'>Upload a video, draw zones, start analysis</span>", unsafe_allow_html=True)
    with col2:
        if st.button("← Back"):
            cleanup()
            st.session_state["zone_frame"] = None
            st.session_state["page"] = "mode"
            st.rerun()

    uploaded = st.file_uploader("Upload MP4 Video", type=["mp4"])
    if uploaded is not None:
        path = "temp_input.mp4"
        with open(path, "wb") as f:
            f.write(uploaded.read())
        if st.session_state["zone_frame"] is None:
            tc = cv2.VideoCapture(path)
            ok, ff = tc.read()
            tc.release()
            if ok:
                st.session_state["zone_frame"] = ff

    if st.session_state["zone_frame"] is not None:
        render_zone_canvas(st.session_state["zone_frame"])

    if st.button("🚀 Process Video") and uploaded is not None:
        app_inst = SurveillanceApp()
        app_inst.set_restricted_zones(st.session_state["zones"])
        st.info("Processing video...")
        summary = app_inst.process_video("temp_input.mp4", "output.mp4")
        st.success("✅ Processing complete!")
        st.write(summary)
        with open("output.mp4", "rb") as f:
            st.download_button("📥 Download Processed Video", f, file_name="processed_video.mp4", use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════

page = st.session_state["page"]
if page == "landing":
    render_landing()
elif page == "mode":
    render_mode()
elif page == "webcam":
    render_webcam()
elif page == "upload":
    render_upload()
