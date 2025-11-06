import numpy as np
import cv2
import mediapipe as mp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from angle_utils import calculate_angle, vertical_angle
# --- INIT ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

stage = None
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        def get_coords(name):
            try:
                lm = landmarks[mp_pose.PoseLandmark[name].value]
                if lm.visibility < 0.5:
                    return None
                return [lm.x * w, lm.y * h]
            except:
                return None

        # Get body parts (with visibility check)
        l_shoulder = get_coords("LEFT_SHOULDER")
        l_elbow = get_coords("LEFT_ELBOW")
        l_wrist = get_coords("LEFT_WRIST")

        r_shoulder = get_coords("RIGHT_SHOULDER")
        r_elbow = get_coords("RIGHT_ELBOW")
        r_wrist = get_coords("RIGHT_WRIST")

        l_hip = get_coords("LEFT_HIP")
        r_hip = get_coords("RIGHT_HIP")

        # Elbow angles
        angles = []
        if l_shoulder and l_elbow and l_wrist:
            left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            angles.append(left_elbow_angle)
        if r_shoulder and r_elbow and r_wrist:
            right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            angles.append(right_elbow_angle)

        avg_elbow_angle = np.mean(angles) if angles else None

        # Spine angle from mid-shoulder to mid-hip (if available)
        mid_shoulder = None
        mid_hip = None
        if l_shoulder and r_shoulder:
            mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        elif l_shoulder:
            mid_shoulder = l_shoulder
        elif r_shoulder:
            mid_shoulder = r_shoulder

        if l_hip and r_hip:
            mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        elif l_hip:
            mid_hip = l_hip
        elif r_hip:
            mid_hip = r_hip

        spine_angle = vertical_angle(mid_shoulder, mid_hip) if mid_shoulder and mid_hip else None

        # Feedback
        feedback = ""
        posture_feedback = ""

        if avg_elbow_angle:
            if stage is None:  # initialize stage
                if avg_elbow_angle > 150:
                    stage = "up"
                elif avg_elbow_angle < 90:
                    stage = "down"

            if avg_elbow_angle > 150:  # Arms extended
                feedback = "Arms fully extended"
                if stage == "down":
                    counter += 1
                    stage = "up"

            elif avg_elbow_angle < 90:  # Bar lowered
                feedback = "Bar at shoulder"
                stage = "down"

            else:
                feedback = "In motion"


        # Display feedback
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f'Reps: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if avg_elbow_angle:
            cv2.putText(frame, f'Elbow: {int(avg_elbow_angle)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if spine_angle is not None:
            cv2.putText(frame, f'Spine Angle: {int(spine_angle)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if feedback:
            cv2.putText(frame, feedback, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if posture_feedback:
            cv2.putText(frame, posture_feedback, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    cv2.imshow("Overhead Press Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
