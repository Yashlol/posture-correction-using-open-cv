import numpy as np

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
    cosine_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


import cv2
import mediapipe as mp

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

        def get_coords(name):
            lm = landmarks[mp_pose.PoseLandmark[name].value]
            return [lm.x * frame.shape[1], lm.y * frame.shape[0]]

        # Get landmarks
        l_shoulder = get_coords("LEFT_SHOULDER")
        l_elbow = get_coords("LEFT_ELBOW")
        l_wrist = get_coords("LEFT_WRIST")

        r_shoulder = get_coords("RIGHT_SHOULDER")
        r_elbow = get_coords("RIGHT_ELBOW")
        r_wrist = get_coords("RIGHT_WRIST")

        l_hip = get_coords("LEFT_HIP")
        r_hip = get_coords("RIGHT_HIP")

        # Calculate angles
        left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

        # Spine angle (avg of shoulders to avg of hips)
        mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]

        spine_angle = vertical_angle(mid_shoulder, mid_hip)

        # Feedback
        feedback = ""
        posture_feedback = ""

        if avg_elbow_angle > 160 and spine_angle < 20:
            feedback = "Arms fully extended"
            posture_feedback = "Good posture"
            if stage == "down":
                counter += 1
                stage = "up"
        elif avg_elbow_angle < 70:
            feedback = "Bar at shoulder"
            stage = "down"
        else:
            feedback = "In motion"

        if 5 <= spine_angle <= 15:
            spine_feedback = "Perfect posture"
        elif 15 < spine_angle <= 25:
            spine_feedback = "Slight lean, adjust if heavy"
        else:
            spine_feedback = "Excessive backward lean"


        # Draw landmarks and display
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f'Reps: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f'Elbow: {int(avg_elbow_angle)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f'Spine Angle: {int(spine_angle)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, feedback, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, posture_feedback, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    cv2.imshow("Overhead Press Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
