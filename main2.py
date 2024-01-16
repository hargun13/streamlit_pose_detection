import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import time 
import threading
# import winsound #this is compatible with windows only not for deployment 
from pydub import AudioSegment
from pydub.playback import play
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

# New variables for user-selected repetition limit and timeout
repetition_limit = 0
# reset_timeout = None

def play_sound():
    sound = AudioSegment.from_wav("sound1.wav")
    play(sound)


# New function to reset counters and update repetition limit
def set_repetition_limit(limit):
    global repetition_limit
    repetition_limit = limit

# New function to reset counters and update repetition limit
def reset_counters():
    global left_counter, right_counter, repetition_limit
    left_counter = 0
    right_counter = 0
    repetition_limit = 0
    # reset_timeout = None

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
            if (angle_left < 50) and left_stage == 'down' and left_prev_stage != 'up':
                left_stage = "up"
                left_counter += 1

            # Curl counter logic for right hand
            if angle_right > 160:
                right_stage = "down"
            if (angle_right < 50) and right_stage == 'down' and right_prev_stage != 'up':
                right_stage = "up"
                right_counter += 1
            
            # Display counters on the frame
            cv2.putText(image, f"Left Reps: {left_counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Right Reps: {right_counter}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Repetition Limit: {repetition_limit}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if left_counter >= repetition_limit and right_counter >= repetition_limit and repetition_limit>0:
                # Exercise completed, play sound and display message

                text_size = cv2.getTextSize("Exercise Completed!", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
                text_x = int((image.shape[1] - text_size[0]) / 2)
                text_y = int((image.shape[0] + text_size[1]) / 2)
                cv2.putText(image, "Exercise Completed!", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                time.sleep(2)
                play_sound()
                
                reset_counters()


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

    hide_st_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    body {
                        margin: 0;
                        padding: 0;
                        background-color: #ffffff;
                    }
                    .st-emotion-cache-1y4p8pa {
                        width: 100%;
                        padding:  0px 10px 0px 10px;
                        max-width: 46rem;
                    }
                    .st-emotion-cache-16txtl3 {
                        padding: 1.5rem 1.5rem;
                    }
                    .st-emotion-cache-1l0ei5a {
                        gap: 0.5rem;
                    }
                    h2 {
                        padding: 15px 0px;
                    }
                    h1 {
                        padding: 0px;
                    }
                    </style>
                    """
    st.markdown(hide_st_style, unsafe_allow_html=True)


    st.sidebar.header("Select Repetition Limit")
    beginner_btn = st.sidebar.button("Beginner - 20 reps", key="beginner")
    intermediate_btn = st.sidebar.button("Intermediate - 30 reps", key="intermediate")
    professional_btn = st.sidebar.button("Professional - 45 reps", key="professional")

    if beginner_btn:
        set_repetition_limit(20)
    elif intermediate_btn:
        set_repetition_limit(30)
    elif professional_btn:
        set_repetition_limit(45)
    
    st.sidebar.header("")
    # Instructions in the sidebar
    st.sidebar.header("Instructions")
    st.sidebar.write("1. Start the camera.")
    st.sidebar.write("2. Select the number of reps you want to do.")
    st.sidebar.write("3. Once you see the number of reps on the screen, continue with the exercise.")
    st.sidebar.write("4. The counter resets itself once the exercise is completed.")


    webrtc_streamer(
        key="bicep-curl-counter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == '__main__':
    main()

# https://bicep-curl-counter.streamlit.app/