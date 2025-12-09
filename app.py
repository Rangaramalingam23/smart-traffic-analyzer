# app.py - Smart Traffic Analyzer (Fixed & Optimized for 2025)
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import uuid

# Page config
st.set_page_config(page_title="Smart Traffic Analyzer", layout="wide")
st.title("Smart City Traffic Analysis System")
st.markdown("### Upload a traffic video → Get real-time vehicle count, congestion level & recommendations")

# Load YOLOv8n model (cached)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Traffic Analyzer Class
class TrafficAnalyzer:
    def __init__(self):
        self.total_vehicles = 0
        self.vehicle_types = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        self.crossed_ids = set()
        self.detection_line_y = None

    def process_frame(self, frame):
        # Use .track() with persist=True for correct tracking IDs
        results = model.track(
            frame,
            persist=True,
            classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
            conf=0.5,
            iou=0.7,
            verbose=False
        )[0]

        if self.detection_line_y is None:
            h, _ = frame.shape[:2]
            self.detection_line_y = h // 2  # Middle line

        for box in results.boxes:
            # Safely get tracking ID (new Ultralytics format)
            track_id = int(box.id.item()) if (hasattr(box, 'id') and box.id is not None) else None

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()

            names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            vehicle_name = names.get(cls, 'unknown')

            # Count vehicle when it crosses the center line (only once
            center_y = (y1 + y2) // 2
            if track_id is not None and center_y > self.detection_line_y and track_id not in self.crossed_ids:
                self.crossed_ids.add(track_id)
                self.total_vehicles += 1
                self.vehicle_types[vehicle_name] += 1

            # Draw bounding box
            color = (0, 255, 0) if cls == 2 else (0, 255, 255) if cls == 5 else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{vehicle_name} {conf:.2f}"
            if track_id is not None:
                label += f" ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw counting line
        cv2.line(frame, (0, self.detection_line_y), (frame.shape[1], self.detection_line_y),
                 (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (10, self.detection_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Overlay stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (580, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Total Vehicles Crossed: {self.total_vehicles}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Cars: {self.vehicle_types['car']}  |  Motorcycles: {self.vehicle_types['motorcycle']}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Buses: {self.vehicle_types['bus']}  |  Trucks: {self.vehicle_types['truck']}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Congestion level
        if self.total_vehicles > 50:
            congestion = "VERY HIGH"
            color = (0, 0, 255)
        elif self.total_vehicles > 30:
            congestion = "HIGH"
            color = (0, 100, 255)
        elif self.total_vehicles > 15:
            congestion = "MEDIUM"
            color = (0, 255, 255)
        else:
            congestion = "LOW"
            color = (0, 255, 0)

        cv2.putText(frame, f"Congestion: {congestion}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        return frame, congestion


# Main app
uploaded_video = st.file_uploader(
    "Upload Traffic Video (MP4, AVI, MOV)",
    type=["mp4", "avi", "mov"],
    key="traffic_video"  # Forces full rerun on new upload
)

if uploaded_video is not None:
    st.video(uploaded_video)

    with st.spinner("Analyzing video... This may take 1-3 minutes depending on length"):
        # Save uploaded video to temp file with unique name
        suffix = os.path.splitext(uploaded_video.name)[1]
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_input.write(uploaded_video.read())
        video_path = temp_input.name
        temp_input.close()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video.")
            os.unlink(video_path)
            st.stop()

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video path
        output_path = f"/tmp/output_{uuid.uuid4().hex}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        analyzer = TrafficAnalyzer()
        progress_bar = st.progress(0)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, _ = analyzer.process_frame(frame)
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 10 == 0 or frame_count == total_frames:
                progress_bar.progress(frame_count / total_frames)

        cap.release()
        out.release()
        progress_bar.empty()

        # Cleanup temp input file
        os.unlink(video_path)

    # Results
    st.success("Analysis Complete!")

    # Show output video
    st.video(output_path)

    # Summary columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Traffic Summary")
        st.write(f"**Total Vehicles Crossed:** {analyzer.total_vehicles}")
        st.write(f"**Cars:** {analyzer.vehicle_types['car']}")
        st.write(f"**Motorcycles:** {analyzer.vehicle_types['motorcycle']}")
        st.write(f"**Buses:** {analyzer.vehicle_types['bus']}")
        st.write(f"**Trucks:** {analyzer.vehicle_types['truck']}")

    with col2:
        st.subheader("Smart Recommendation")
        if analyzer.total_vehicles > 50:
            st.error("VERY HIGH CONGESTION! Deploy traffic police + adjust signals urgently!")
        elif analyzer.total_vehicles > 30:
            st.warning("High traffic — Consider dynamic lane reversal or signal optimization.")
        elif analyzer.total_vehicles > 15:
            st.info("Moderate traffic flow — Optimize signal timing recommended.")
        else:
            st.success("Smooth traffic — All good!")

    # Download button
    with open(output_path, "rb") as video_file:
        st.download_button(
            label="Download Processed Video",
            data=video_file,
            file_name="smart_traffic_analysis_result.mp4",
            mime="video/mp4"
        )

    # Clean up output file after download
    if os.path.exists(output_path):
        os.remove(output_path)

else:
    st.info("Please upload a traffic video to begin analysis.")
    st.markdown("""
    **Supported formats:** MP4, AVI, MOV  
    **Best results:** Clear daylight footage, camera looking down the road (not sideways)
    """)