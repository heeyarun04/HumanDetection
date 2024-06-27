import streamlit as st
import logging
import cv2
import torch
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

# Configure logging
logger = logging.getLogger("streamlit_webrtc")
logger.setLevel(logging.DEBUG)

# Stream handler for console output
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load YOLOv8 model
model = YOLO('yolov8m.pt')  # Use a pre-trained YOLOv8 model

# App title
st.title("Human Detection as Indicator Automation")

# Streamlit WebRTC configuration
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=None,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

# Logging initialization message
logger.info("WebRTC streamer initialized.")

if webrtc_ctx.video_receiver:
    # Initialize the VideoCapture object for IP camera
    cap = cv2.VideoCapture("192.168.43.254")
    logger.info("VideoCapture object created for IP camera.")

    while True:
        # Read frame from IP camera
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from IP camera.")
            break

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results.xyxy[0]:  # x1, y1, x2, y2, confidence, class
            x1, y1, x2, y2, conf, cls = result
            if int(cls) == 0:  # Class 0 is 'person' in COCO dataset
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f'Person: {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                logger.info(f"Human detected at [{x1}, {y1}, {x2}, {y2}] with confidence {conf:.2f}")

        # Display the frame in Streamlit
        webrtc_ctx.video_receiver.process_frame(frame)
        st.image(frame, channels="BGR")

    # Release the VideoCapture object and cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("VideoCapture object released and resources cleaned up.")
else:
    logger.warning("No video receiver available.")

