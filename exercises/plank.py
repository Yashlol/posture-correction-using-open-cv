import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from angle_utils import calculate_angle, vertical_angle, horizontal_alignment

# --- INIT ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

start_time = 0
plank_timer = 0
form_good = False
bad_form_start = None

cv2.namedWindow('Plank Form Tracker', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Plank Form Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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
        lm = results.pose_landmarks.landmark

        def get_point(name):
            pt = lm[mp_pose.PoseLandmark[name].value]
            return [pt.x, pt.y], pt.visibility

        # Get points and visibility
        (l_shoulder, l_vis), (r_shoulder, r_vis) = get_point("LEFT_SHOULDER"), get_point("RIGHT_SHOULDER")
        (l_elbow, le_vis), (r_elbow, re_vis) = get_point("LEFT_ELBOW"), get_point("RIGHT_ELBOW")
        (l_wrist, lw_vis), (r_wrist, rw_vis) = get_point("LEFT_WRIST"), get_point("RIGHT_WRIST")
        (l_hip, lh_vis), (r_hip, rh_vis) = get_point("LEFT_HIP"), get_point("RIGHT_HIP")
        (l_knee, lk_vis), (r_knee, rk_vis) = get_point("LEFT_KNEE"), get_point("RIGHT_KNEE")
        (l_ankle, la_vis), (r_ankle, ra_vis) = get_point("LEFT_ANKLE"), get_point("RIGHT_ANKLE")

        # Angles
        spine_angle = vertical_angle(l_shoulder, l_hip)
        hip_diff = horizontal_alignment(l_hip, r_hip)

        # Default to None
        l_elbow_angle = r_elbow_angle = l_knee_angle = r_knee_angle = None

        # Check visibility and calculate only if available
        if l_vis > 0.5 and le_vis > 0.5 and lw_vis > 0.5:
            l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        if r_vis > 0.5 and re_vis > 0.5 and rw_vis > 0.5:
            r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        if lh_vis > 0.5 and lk_vis > 0.5 and la_vis > 0.5:
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        if rh_vis > 0.5 and rk_vis > 0.5 and ra_vis > 0.5:
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # --- FORM LOGIC ---
        is_good_form = False

        if all(x is not None for x in [l_elbow_angle, r_elbow_angle, l_knee_angle, r_knee_angle]):
            is_good_form = (
                spine_angle < 10 and
                80 < l_elbow_angle < 100 and
                80 < r_elbow_angle < 100 and
                160 < l_knee_angle < 180 and
                160 < r_knee_angle < 180 and
                hip_diff < 5
            )
        elif all(x is not None for x in [l_elbow_angle, l_knee_angle]):
            is_good_form = (
                spine_angle < 10 and
                80 < l_elbow_angle < 100 and
                160 < l_knee_angle < 180 and
                hip_diff < 5
            )
        elif all(x is not None for x in [r_elbow_angle, r_knee_angle]):
            is_good_form = (
                spine_angle < 10 and
                80 < r_elbow_angle < 100 and
                160 < r_knee_angle < 180 and
                hip_diff < 5
            )

        current_time = time.time()

        if is_good_form:
            if not form_good:
                form_good = True
                bad_form_start = None
                start_time = current_time  # Start timer from this point
            else:
                plank_timer += current_time - start_time
                start_time = current_time  # Update start time for next frame
        else:
            if form_good:
                if bad_form_start is None:
                    bad_form_start = current_time
                elif current_time - bad_form_start >= 3:
                    form_good = False
                    bad_form_start = None
            if not form_good:
                start_time = current_time  # Don't reset timer, just update to avoid extra time


        # DISPLAY
        cv2.putText(image, f'Timer: {int(plank_timer)}s', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(image, f'Spine: {int(spine_angle)}', (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f'L Elbow: {int(l_elbow_angle) if l_elbow_angle else 0}', (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
        cv2.putText(image, f'R Elbow: {int(r_elbow_angle) if r_elbow_angle else 0}', (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(image, f'L Knee: {int(l_knee_angle) if l_knee_angle else 0}', (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(image, f'R Knee: {int(r_knee_angle) if r_knee_angle else 0}', (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 0), 2)
        cv2.putText(image, f'Hip Alignment: {hip_diff:.2f}%', (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

        if is_good_form:
            cv2.putText(image, "Form: Good", (30, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Form: Bad", (30, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except Exception as e:
        cv2.putText(image, "No pose detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Plank Form Tracker', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()