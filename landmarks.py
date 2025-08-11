import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1)  # Lower complexity for speed

def outline(frame):
    results = pose.process(frame)
    return results.pose_landmarks

