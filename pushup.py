import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

rep_count = 0
pushup_state = "up"

# Fullscreen setup
cv2.namedWindow('Pushup Tracker', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Pushup Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

            # Landmarks
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            alignment_angle = calculate_angle(shoulder, hip, ankle)
            knee_angle = calculate_angle(hip, knee, ankle)

            # Feedback
            if alignment_angle > 170:
                form_feedback = "Good posture"
            else:
                form_feedback = "Keep your body straight"

            if knee_angle < 170:
                knee_feedback = "Straighten your knees"
            else:
                knee_feedback = "Knees straight"

            # Rep counter logic
            if elbow_angle > 160:
                pushup_state = "up"
            if (
                elbow_angle < 90
                and pushup_state == "up"
                and alignment_angle > 160
                and knee_angle > 170
            ):
                pushup_state = "down"
                rep_count += 1

            # Display
            cv2.putText(image, f'Elbow: {int(elbow_angle)}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(image, f'Spine: {int(alignment_angle)}', (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f'Knee: {int(knee_angle)}', (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
            cv2.putText(image, form_feedback, (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)
            cv2.putText(image, knee_feedback, (30, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(image, f'Reps: {rep_count}', (30, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Pushup Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
