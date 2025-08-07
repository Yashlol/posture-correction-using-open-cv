import cv2
import mediapipe as mp
import numpy as np
import time

# --- ANGLE FUNCTIONS ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def horizontal_angle(a, b):
    a, b = np.array(a), np.array(b)
    vector = a - b
    horizontal = np.array([1, 0])
    cosine = np.dot(vector, horizontal) / (np.linalg.norm(vector) * np.linalg.norm(horizontal))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# --- INIT ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

start_time = 0
hold_time = 0
form_good = False
bad_form_timer = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            # Get required joint coordinates
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            spine_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            hip_level_angle = horizontal_angle(l_hip, r_hip)

            # Check form conditions
            good_spine = 160 <= spine_angle <= 200
            good_elbows = 70 <= l_elbow_angle <= 110 and 70 <= r_elbow_angle <= 110
            good_knees = 160 <= l_knee_angle <= 200 and 160 <= r_knee_angle <= 200
            good_hips = hip_level_angle <= 10

            all_good = good_spine and good_elbows and good_knees and good_hips

            if all_good:
                if not form_good:
                    form_good = True
                    bad_form_timer = None
                    start_time = time.time() - hold_time  # resume timer
                else:
                    hold_time = time.time() - start_time
            else:
                if form_good:
                    if bad_form_timer is None:
                        bad_form_timer = time.time()
                    elif time.time() - bad_form_timer > 3:
                        form_good = False
                        hold_time = time.time() - start_time  # pause timer

            # Display angles and timer
            cv2.putText(image, f'Hold Time: {int(hold_time)} sec', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if form_good else (0, 0, 255), 2)

            cv2.putText(image, f'Spine: {int(spine_angle)}', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image, f'L Elbow: {int(l_elbow_angle)}  R Elbow: {int(r_elbow_angle)}', (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
            cv2.putText(image, f'L Knee: {int(l_knee_angle)}  R Knee: {int(r_knee_angle)}', (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            cv2.putText(image, f'Hip Level: {int(hip_level_angle)}', (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Plank Form Checker with Timer', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
