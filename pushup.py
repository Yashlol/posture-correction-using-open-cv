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

            # LEFT SIDE
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # RIGHT SIDE
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # ANGLES
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            spine_angle_left = calculate_angle(left_shoulder, left_hip, left_ankle)
            spine_angle_right = calculate_angle(right_shoulder, right_hip, right_ankle)
            avg_spine_angle = (spine_angle_left + spine_angle_right) / 2

           
            # Relaxed pushup rep criteria:
            elbows_down = left_elbow_angle < 100 and right_elbow_angle < 100
            elbows_up = left_elbow_angle > 150 and right_elbow_angle > 150

            knees_straight = left_knee_angle > 160 and right_knee_angle > 160
            body_straight = avg_spine_angle > 150

            # Feedback
            form_feedback = "Good posture" if body_straight else "Keep your body straight"
            knee_feedback = "Knees straight" if knees_straight else "Straighten your knees"
            elbow_feedback = "Lower down" if not elbows_down else "Good depth"

            # REP COUNTER
            if elbows_down and body_straight and knees_straight:
                if pushup_state == "up":
                    pushup_state = "down"

            if elbows_up and pushup_state == "down":
                pushup_state = "up"
                rep_count += 1

            # DISPLAY INFO
            cv2.putText(image, f'Left Elbow: {int(left_elbow_angle)}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(image, f'Right Elbow: {int(right_elbow_angle)}', (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(image, f'Spine: {int(avg_spine_angle)}', (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(image, f'Left Knee: {int(left_knee_angle)}', (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
            cv2.putText(image, f'Right Knee: {int(right_knee_angle)}', (30, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)

            cv2.putText(image, form_feedback, (30, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)
            cv2.putText(image, knee_feedback, (30, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(image, elbow_feedback, (30, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.putText(image, f'Reps: {rep_count}', (30, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Pushup Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
