import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import math

st.title("AuraWork Guardian")
st.write("กดปุ่ม START ด้านล่างเพื่อเปิดกล้องตรวจจับท่านั่ง")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    ang = abs(ang)
    return 360 - ang if ang > 180 else ang

class PostureTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        status_text = "Good Posture"
        status_color = (100, 230, 100)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            ear = [lm[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, lm[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            hip = [shoulder[0], shoulder[1] + 100]

            if calculate_angle(ear, shoulder, hip) < 145:
                status_text = "Time to Reset Your Neck"
                status_color = (50, 180, 255)
                cv2.rectangle(img, (0, 0), (w, h), (50, 180, 255), 12)

        cv2.putText(img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        return img

webrtc_streamer(key="posture", video_transformer_factory=PostureTransformer)