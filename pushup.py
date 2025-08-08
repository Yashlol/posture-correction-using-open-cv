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

cv2.namedWindow('Pushup Tracker', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Pushup Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
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

        form_feedback = knee_feedback = elbow_feedback = ""
        left_elbow_angle = right_elbow_angle = 0
        left_knee_angle = right_knee_angle = 0
        avg_spine_angle = 0

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark

                def get_landmark(name):
                    lm = landmarks[mp_pose.PoseLandmark[name].value]
                    return [lm.x, lm.y], lm.visibility

                # Left side
                left_shoulder, v1 = get_landmark("LEFT_SHOULDER")
                left_elbow, v2 = get_landmark("LEFT_ELBOW")
                left_wrist, v3 = get_landmark("LEFT_WRIST")
                left_hip, v4 = get_landmark("LEFT_HIP")
                left_knee, v5 = get_landmark("LEFT_KNEE")
                left_ankle, v6 = get_landmark("LEFT_ANKLE")

                # Right side
                right_shoulder, v7 = get_landmark("RIGHT_SHOULDER")
                right_elbow, v8 = get_landmark("RIGHT_ELBOW")
                right_wrist, v9 = get_landmark("RIGHT_WRIST")
                right_hip, v10 = get_landmark("RIGHT_HIP")
                right_knee, v11 = get_landmark("RIGHT_KNEE")
                right_ankle, v12 = get_landmark("RIGHT_ANKLE")

                left_visible = all(v > 0.6 for v in [v1, v2, v3, v4, v5, v6])
                right_visible = all(v > 0.6 for v in [v7, v8, v9, v10, v11, v12])

                if left_visible and right_visible:
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    spine_angle_left = calculate_angle(left_shoulder, left_hip, left_ankle)
                    spine_angle_right = calculate_angle(right_shoulder, right_hip, right_ankle)
                    avg_spine_angle = (spine_angle_left + spine_angle_right) / 2
                elif left_visible:
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    avg_spine_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
                    right_elbow_angle = left_elbow_angle
                    right_knee_angle = left_knee_angle
                elif right_visible:
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    avg_spine_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
                    left_elbow_angle = right_elbow_angle
                    left_knee_angle = right_knee_angle
                else:
                    raise ValueError("Object not in frame")

                # Form checks
                elbows_down = left_elbow_angle < 100 and right_elbow_angle < 100
                elbows_up = left_elbow_angle > 150 and right_elbow_angle > 150
                knees_straight = left_knee_angle > 160 and right_knee_angle > 160
                body_straight = avg_spine_angle > 150

                # Feedback
                form_feedback = "Good posture" if body_straight else "Keep your body straight"
                knee_feedback = "Knees straight" if knees_straight else "Straighten your knees"
                elbow_feedback = "Lower down" if not elbows_down else "Good depth"

                # State update
                if elbows_down and body_straight and knees_straight and pushup_state == "up":
                    pushup_state = "down"
                elif elbows_up and pushup_state == "down":
                    pushup_state = "up"
                    rep_count += 1

            except ValueError as ve:
                form_feedback = str(ve)
                knee_feedback = ""
                elbow_feedback = ""

        else:
            form_feedback = "Object not in frame"
            knee_feedback = ""
            elbow_feedback = ""


        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw Text
        cv2.putText(image, f'Left Elbow: {int(left_elbow_angle)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f'Right Elbow: {int(right_elbow_angle)}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f'Spine: {int(avg_spine_angle)}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(image, f'Left Knee: {int(left_knee_angle)}', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
        cv2.putText(image, f'Right Knee: {int(right_knee_angle)}', (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)

        cv2.putText(image, form_feedback, (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)
        cv2.putText(image, knee_feedback, (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(image, elbow_feedback, (30, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(image, f'Reps: {rep_count}', (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow('Pushup Tracker', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
