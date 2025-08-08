# Updated squat.py with dynamic side check
import cv2
import mediapipe as mp
import numpy as np

# --- ANGLE FUNCTIONS ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def vertical_angle(a, b):
    a, b = np.array(a), np.array(b)
    vector = a - b
    vertical = np.array([0, -1])
    cosine = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# --- INIT ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

rep_count = 0
squat_state = "up"

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=2,
                  enable_segmentation=False,
                  min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:
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
                    spine_angle = vertical_angle(shoulder, hip)

                    side_angles.append(knee_angle)
                    side_spines.append(spine_angle)

                except:
                    continue

            if len(side_angles) == 0:
                raise ValueError("No side fully visible")

            avg_knee_angle = sum(side_angles) / len(side_angles)
            avg_spine_angle = sum(side_spines) / len(side_spines)

            # FEEDBACK
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

            # REP LOGIC
            if squat_feedback == "Good squat" and spine_feedback == "Correct posture":
                if squat_state == "up":
                    squat_state = "down"

            if squat_feedback == "Standing tall" and spine_feedback == "Upright posture":
                if squat_state == "down":
                    rep_count += 1
                    squat_state = "up"

            # DISPLAY
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
