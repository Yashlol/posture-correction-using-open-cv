import cv2
import mediapipe as mp
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from angle_utils import calculate_angle, vertical_angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

rep_count = 0
squat_state = "up"

# --- Fullscreen window setup ---
cv2.namedWindow('Squat Form Correction + Rep Counter', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Squat Form Correction + Rep Counter', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # LEFT SIDE
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # RIGHT SIDE
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # ANGLES
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            l_spine_angle = vertical_angle(l_shoulder, l_hip)
            r_spine_angle = vertical_angle(r_shoulder, r_hip)

            # AVERAGE FOR CONSISTENT FEEDBACK
            avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
            avg_spine_angle = (l_spine_angle + r_spine_angle) / 2

            # FEEDBACK
            if avg_knee_angle > 140:
                squat_feedback = "Standing tall"
            elif avg_knee_angle > 80:
                squat_feedback = "Good squat"
            elif avg_knee_angle < 30:
                squat_feedback = "Too low"

            if avg_spine_angle < 10:
                spine_feedback = "Upright posture"
            elif avg_spine_angle < 30:
                spine_feedback = "Correct posture"
            else:
                spine_feedback = "Too much forward lean"

            # --- REP LOGIC ---
            if squat_feedback == "Good squat" and spine_feedback == "Correct posture":
                if squat_state == "up":
                    squat_state = "down"  # Going down with good form

            if squat_feedback == "Standing tall" and spine_feedback == "Upright posture":
                if squat_state == "down":
                    rep_count += 1
                    squat_state = "up"  # Completed one rep

            # --- DISPLAY ---
            cv2.putText(image, f'Reps: {rep_count}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(image, f'Knee Angle: {int(avg_knee_angle)}', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(image, f'Spine Lean: {int(avg_spine_angle)}', (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, squat_feedback, (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(image, spine_feedback, (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Squat Form Correction + Rep Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
