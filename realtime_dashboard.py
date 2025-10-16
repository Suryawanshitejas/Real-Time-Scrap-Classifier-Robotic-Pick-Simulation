import streamlit as st
import pandas as pd
import cv2
from ultralytics import YOLO
from datetime import datetime
from collections import Counter, deque
import time

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "runs/detect/train2/weights/best.pt"  # Update path if needed
model = YOLO(MODEL_PATH)

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Scrap Material Detection", layout="wide")
st.title("♻️ Real-Time Scrap Classifier & Robotic Pick Simulation")

# Layout
col1, col2 = st.columns([3, 2])

# Placeholders
frame_placeholder = col1.empty()
chart_placeholder = col2.empty()
log_placeholder = st.empty()
item_placeholder = col1.empty()

# ---------------------------
# Initialize
# ---------------------------
counts = Counter()
log_deque = deque(maxlen=50)  # last 50 detections

# Control start/stop detection
if "run" not in st.session_state:
    st.session_state.run = False

start_btn, stop_btn = st.columns(2)
with start_btn:
    if st.button("▶️ Start Detection"):
        st.session_state.run = True
with stop_btn:
    if st.button("🛑 Stop Detection"):
        st.session_state.run = False

# ---------------------------
# Open Webcam
# ---------------------------
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

cap = st.session_state.cap
if not cap.isOpened():
    st.error("❌ Cannot access webcam.")
    st.stop()

# ---------------------------
# Detection Loop
# ---------------------------
while st.session_state.run and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Failed to capture frame.")
        break

    # YOLO inference
    results = model(frame, stream=True)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detected_items = []

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            counts[label] += 1
            detected_items.append(label)

            # Log detection
            log_deque.append([current_time, label, conf])

            # Draw box on frame (optional)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Convert frame for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Show detected items
    if detected_items:
        item_placeholder.markdown("**Detected Items:** " + ", ".join(detected_items))

    # Show bar chart
    if counts:
        chart_df = pd.DataFrame({
            "Material": list(counts.keys()),
            "Count": list(counts.values())
        })
        chart_placeholder.bar_chart(chart_df.set_index("Material"))

    # Show recent logs
    if log_deque:
        df = pd.DataFrame(log_deque, columns=["timestamp", "class", "confidence"])
        log_placeholder.dataframe(df.tail(10), use_container_width=True)

    # Short delay for performance
    time.sleep(0.1)

# Release webcam when stopped
if not st.session_state.run:
    cap.release()
    st.success("✅ Detection stopped and camera released.")
