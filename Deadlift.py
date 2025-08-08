
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
lift_state = "up"

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

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            spine_angle = vertical_angle(shoulder, hip)

            # Feedback logic
            if knee_angle > 160 and hip_angle > 160:
                lift_feedback = "Standing tall"
            elif 90 < knee_angle < 140 and 40 < hip_angle < 80:
                lift_feedback = "Proper deadlift position"
            else:
                lift_feedback = "Fix your form"

            if lift_feedback == "Proper deadlift position" and lift_state == "up":
                lift_state = "down"

            if lift_feedback == "Standing tall" and lift_state == "down":
                rep_count += 1
                lift_state = "up"

            # Display on screen
            cv2.putText(image, f'Reps: {rep_count}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(image, f'Knee Angle: {int(knee_angle)}', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(image, f'Hip Hinge Angle: {int(hip_angle)}', (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
            cv2.putText(image, f'Spine Lean: {int(spine_angle)}', (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(image, lift_feedback, (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Deadlift Form Checker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
