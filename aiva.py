import streamlit as st
import cv2
import pandas as pd
import mediapipe as mp
from deepface import DeepFace
from datetime import datetime
import tempfile
import pyngrok.ngrok

# Title of the Streamlit app
st.title("AI-Based Office Monitoring System")

# Initialize variables
in_time, out_time = {}, {}
activity_log = []
sentiment_log = []
idle_time = None

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Function to analyze sentiment using DeepFace
def analyze_sentiment(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result['dominant_emotion']
    except:
        return "Unknown"

# Video capture and processing
def video_processing():
    cap = cv2.VideoCapture(0)  # For webcam, change to a video file path if needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        faces = face_detection.process(rgb_frame).detections
        current_time = datetime.now().strftime('%H:%M:%S')

        if faces:
            for face in faces:
                # Placeholder for actual recognition (Here, we'll use a dummy name)
                employee_name = "Employee"
                
                # Log in-time
                if employee_name not in in_time:
                    in_time[employee_name] = current_time

                # Sentiment analysis
                sentiment = analyze_sentiment(rgb_frame)
                sentiment_log.append({'time': current_time, 'employee': employee_name, 'sentiment': sentiment})

                # Display results
                st.write(f"Employee: {employee_name}, Sentiment: {sentiment}, Time: {current_time}")
        else:
            st.write("No employees detected. Room is idle.")
        
        st.image(frame, channels="BGR")

    cap.release()

# Streamlit UI
if st.button("Start Monitoring"):
    video_processing()
    
st.write("Monitoring stopped.")

# Display logs
if sentiment_log:
    st.write("Sentiment Log:")
    st.dataframe(pd.DataFrame(sentiment_log))
