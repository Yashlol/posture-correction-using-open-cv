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

            # LANDMARKS
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # ANGLES
            knee_angle = calculate_angle(hip, knee, ankle)
            spine_angle = vertical_angle(shoulder, hip)

            # FEEDBACK
            if knee_angle > 140:
                squat_feedback = "Standing tall"
            elif knee_angle < 60:
                squat_feedback = "Too low"
            else:
                squat_feedback = "Good squat"

            if spine_angle < 10:
                spine_feedback = "Upright posture"
            elif spine_angle < 30:
                spine_feedback = "correct posture"
            else:
                spine_feedback = "Too much forward lean"

            # --- REP LOGIC ---
            if squat_feedback == "Good squat" and spine_feedback == "correct posture":
                if squat_state == "up":
                    squat_state = "down"  # Going down with good form

            if squat_feedback == "Standing tall" and spine_feedback == "Upright posture":
                if squat_state == "down":
                    rep_count += 1
                    squat_state = "up"  # Completed one rep

            # --- DISPLAY ---
            cv2.putText(image, f'Reps: {rep_count}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(image, f'Knee: {int(knee_angle)}', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(image, f'Spine Lean: {int(spine_angle)}', (30, 120),
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


# | **Angle**       | **Good Range** | **What it Indicates**                            |
# | --------------- | -------------- | ------------------------------------------------ |
# | **Knee Angle**  | 60°–140°       | Indicates depth of squat                         |
# | **Spine Angle** | 0°–10°         | Upright spine posture                            |
# |                 | 10°–30°        | Acceptable forward lean (depending on body type) |
# |                 | >30°           | Excessive lean — potential form issue            |
