# squat.py - Cleaner version using helper modules

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera import start_camera
from landmarks import outline
from angle_utils import calculate_angle, calculate_vertical_angle
from counter import Counter
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize camera and rep counter
cap = start_camera()
rep_counter = Counter()

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,  # Reduced for less lag
                  enable_segmentation=False,
                  min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:

    squat_state = "up"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = outline(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if landmarks:
            side_angles = []
            side_spines = []

            for side in ["LEFT", "RIGHT"]:
                try:
                    shoulder = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x,
                                landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y]
                    hip = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x,
                           landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y]
                    knee = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x,
                            landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].y]
                    ankle = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x,
                             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].y]

                    knee_angle = calculate_angle(hip, knee, ankle)
                    spine_angle = calculate_vertical_angle(shoulder, hip)

                    side_angles.append(knee_angle)
                    side_spines.append(spine_angle)
                except:
                    continue

            if side_angles:
                avg_knee_angle = sum(side_angles) / len(side_angles)
                avg_spine_angle = sum(side_spines) / len(side_spines)

                # Feedback
                if avg_knee_angle > 140:
                    squat_feedback = "Standing tall"
                elif avg_knee_angle < 60:
                    squat_feedback = "Too low"
                else:
                    squat_feedback = "Good squat"

                if avg_spine_angle < 10:
                    spine_feedback = "Upright posture"
                elif avg_spine_angle < 30:
                    spine_feedback = "Correct posture"
                else:
                    spine_feedback = "Too much forward lean"

                # Rep Logic
                if squat_feedback == "Good squat" and spine_feedback == "Correct posture":
                    if squat_state == "up":
                        squat_state = "down"

                if squat_feedback == "Standing tall" and spine_feedback == "Upright posture":
                    if squat_state == "down":
                        rep_counter.increment()
                        squat_state = "up"

                # Display info
                cv2.putText(frame, f'Reps: {rep_counter.get_count()}', (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f'Knee Angle: {int(avg_knee_angle)}', (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame, f'Spine Lean: {int(avg_spine_angle)}', (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, squat_feedback, (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, spine_feedback, (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)

            else:
                cv2.putText(frame, "Body not fully visible", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "No person in frame", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Squat Form Correction + Rep Counter', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
