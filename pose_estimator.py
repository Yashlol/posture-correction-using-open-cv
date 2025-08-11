import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def get_landmarks(frame, draw=False):
    """
    Processes a video frame and returns pose landmarks.
    
    Args:
        frame (np.array): BGR image from OpenCV.
        draw (bool): Whether to draw landmarks on the frame.

    Returns:
        tuple: (results, frame_with_landmarks)
            - results: MediaPipe Pose results object
            - frame_with_landmarks: Frame with landmarks drawn (if draw=True)
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if draw and results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )
    
    return results, frame
