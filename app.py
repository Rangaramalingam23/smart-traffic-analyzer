# app.py - Smart Traffic Analysis System (Live Web App)
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd

# Page config
st.set_page_config(page_title="Smart Traffic Analyzer", layout="wide")
st.title("Smart City Traffic Analysis System")
st.markdown("### Upload a traffic video → Get instant vehicle count, congestion level & smart recommendations")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Traffic Analyzer Class (simplified & optimized for web)
class TrafficAnalyzer:
    def __init__(self):
        self.total_vehicles = 0
        self.vehicle_types = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        self.crossed_ids = set()
        self.detection_line_y = None

    def process_frame(self, frame):
        results = model(frame, classes=[2,3,5,7], conf=0.5)[0]  # car, motorcycle, bus, truck
        detections = []
        
        if self.detection_line_y is None:
            h, w = frame.shape[:2]
            self.detection_line_y = h // 2

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            # Vehicle type mapping
            names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            vehicle_name = names.get(cls, 'unknown')

            # Count only when vehicle crosses the middle line
            center_y = (y1 + y2) // 2
            if center_y > self.detection_line_y and track_id not in self.crossed_ids:
                if track_id != -1:
                    self.crossed_ids.add(track_id)
                    self.total_vehicles += 1
                    self.vehicle_types[vehicle_name] += 1

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': vehicle_name,
                'conf': conf
            })

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_name} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw detection line
        cv2.line(frame, (0, self.detection_line_y), (frame.shape[1], self.detection_line_y), (255, 255, 0), 3)
        cv2.putText(frame, "COUNTING LINE", (10, self.detection_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Overlay stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"Total Vehicles Crossed: {self.total_vehicles}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Cars: {self.vehicle_types['car']} | Bikes: {self.vehicle_types['motorcycle']}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Buses: {self.vehicle_types['bus']} | Trucks: {self.vehicle_types['truck']}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        congestion = "LOW"
        if self.total_vehicles > 50: congestion = "VERY HIGH"
        elif self.total_vehicles > 30: congestion = "HIGH"
        elif self.total_vehicles > 15: congestion = "MEDIUM"
        color = (0, 255, 0) if congestion in ["LOW", "MEDIUM"] else (0, 0, 255)
        cv2.putText(frame, f"Congestion: {congestion}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        return frame, congestion

# File uploader
uploaded_video = st.file_uploader("Upload Traffic CCTV Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.video(uploaded_video)
    
    with st.spinner("Analyzing traffic video... This may take 1-3 minutes"):
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video
        output_path = "output_traffic.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        analyzer = TrafficAnalyzer()
        frame_count = 0
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, congestion = analyzer.process_frame(frame)
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 10 == 0:
                progress_bar.progress(frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        cap.release()
        out.release()
        progress_bar.empty()

    # Results
    st.success("Analysis Complete!")
    st.video(output_path)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Traffic Summary")
        st.write(f"**Total Vehicles Crossed:** {analyzer.total_vehicles}")
        st.write(f"**Cars:** {analyzer.vehicle_types['car']}")
        st.write(f"**Motorcycles:** {analyzer.vehicle_types['motorcycle']}")
        st.write(f"**Buses:** {analyzer.vehicle_types['bus']}")
        st.write(f"**Trucks:** {analyzer.vehicle_types['truck']}")

    with col2:
        st.subheader("Smart City Recommendation")
        if analyzer.total_vehicles > 50:
            st.error("VERY HIGH CONGESTION! Deploy traffic police + adjust signals!")
        elif analyzer.total_vehicles > 30:
            st.warning("High traffic. Consider dynamic lane reversal.")
        elif analyzer.total_vehicles > 15:
            st.info("Moderate flow. Optimize signal timing.")
        else:
            st.success("Smooth traffic. All good!")

    # Download
    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="traffic_analysis_result.mp4")