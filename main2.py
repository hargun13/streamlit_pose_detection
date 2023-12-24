import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from sample_utils.turn import get_ice_servers

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Curl counter variables for left and right hands
left_counter = 0
left_stage = None
left_prev_stage = None

right_counter = 0
right_stage = None
right_prev_stage = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def process_frame(frame):
    global left_counter, right_counter, left_stage, left_prev_stage, right_stage, right_prev_stage

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # Get coordinates for left hand
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get coordinates for right hand
            shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles for left and right hands
            angle_left = calculate_angle(shoulder, elbow, wrist)
            angle_right = calculate_angle(shoulder1, elbow1, wrist1)

            # Curl counter logic for left hand
            if angle_left > 160:
                left_stage = "down"
            if (angle_left < 45) and left_stage == 'down' and left_prev_stage != 'up':
                left_stage = "up"
                left_counter += 1

            # Curl counter logic for right hand
            if angle_right > 160:
                right_stage = "down"
            if (angle_right < 45) and right_stage == 'down' and right_prev_stage != 'up':
                right_stage = "up"
                right_counter += 1

            left_prev_stage = left_stage
            right_prev_stage = right_stage

        except Exception as e:
            print(f"Exception: {e}")

        return image

def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    processed_frame = process_frame(img)
    return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

def main():
    st.title("Bicep Curl Counter")

    webrtc_streamer(
        key="bicep-curl-counter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.text(f"Left Reps: {left_counter} | Right Reps: {right_counter}")

if __name__ == '__main__':
    main()
