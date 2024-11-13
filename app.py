import cv2
import numpy as np
import streamlit as st
from detect import detect_gender_age  # Assuming you have this in detect.py

# Setup Streamlit page layout
st.title("Real-Time Gender and Age Detection")
st.write("Detect gender and age from webcam feed")

# Open webcam feed using OpenCV
cap = cv2.VideoCapture(0)

# Placeholder for displaying the video
frame_placeholder = st.empty()

# Placeholder for the output text (gender, age)
output_placeholder = st.empty()

# We want to continuously update the output in the same place
while True:
    ret, frame = cap.read()  # Capture a frame from the webcam feed

    if not ret:
        break

    # Process the frame to detect gender and age
    processed_frame, results = detect_gender_age(frame)

    # Convert the processed frame to RGB for Streamlit display
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Show the processed frame in the placeholder
    frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

    # Update the gender and age output in the same place
    if results:
        output_placeholder.empty()  # Clear the previous output
        output_placeholder.write(f"Detected: {results[0]}")  # Display the first result

    # Break the loop on 'q' key press (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
