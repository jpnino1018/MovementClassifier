import os
import cv2
import json
from tqdm import tqdm
import mediapipe as mp
import numpy as np

# === Setup ===
VIDEOS_DIR = 'videos'
OUTPUT_DIR = 'labeled_landmarks'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_xyz(landmarks, idx):
    lm = landmarks[idx.value]
    return [lm.x, lm.y, lm.z]

def avg_xyz(landmarks, idx1, idx2):
    l1, l2 = landmarks[idx1.value], landmarks[idx2.value]
    return [(l1.x + l2.x)/2, (l1.y + l2.y)/2, (l1.z + l2.z)/2]

# === Loop through all folders and videos ===
for label in os.listdir(VIDEOS_DIR):
    label_path = os.path.join(VIDEOS_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for video_file in os.listdir(label_path):
        if not video_file.endswith(('.mp4', '.MOV', '.avi')):
            continue

        video_path = os.path.join(label_path, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        output = []

        for frame_num in tqdm(range(frame_count), desc=f"{label}/{video_file}"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Extract landmarks for angle calculation
                left_shoulder = get_xyz(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
                right_shoulder = get_xyz(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                mid_shoulder = avg_xyz(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)

                left_hip = get_xyz(lm, mp_pose.PoseLandmark.LEFT_HIP)
                right_hip = get_xyz(lm, mp_pose.PoseLandmark.RIGHT_HIP)
                mid_hip = avg_xyz(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)

                left_knee = get_xyz(lm, mp_pose.PoseLandmark.LEFT_KNEE)
                right_knee = get_xyz(lm, mp_pose.PoseLandmark.RIGHT_KNEE)

                left_ankle = get_xyz(lm, mp_pose.PoseLandmark.LEFT_ANKLE)
                right_ankle = get_xyz(lm, mp_pose.PoseLandmark.RIGHT_ANKLE)

                # Calculate angles
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                
                # Trunk inclination (angle of hips-shoulders line with vertical)
                # We can approximate this by using the angle with a horizontal line
                vertical_ref_point = [mid_hip[0], mid_hip[1] - 1] # A point directly above the hip
                trunk_inclination = calculate_angle(vertical_ref_point, mid_hip, mid_shoulder)

                output.append({
                    "frame": frame_num,
                    "timestamp": round(frame_num / fps, 2),
                    "landmarks": {
                        "head": get_xyz(lm, mp_pose.PoseLandmark.NOSE),
                        "shoulders": avg_xyz(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                        "wrists": avg_xyz(lm, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST),
                        "hips": avg_xyz(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                        "knees": avg_xyz(lm, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE),
                        "ankles": avg_xyz(lm, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    },
                    "features": {
                        "left_knee_angle": left_knee_angle,
                        "right_knee_angle": right_knee_angle,
                        "left_hip_angle": left_hip_angle,
                        "right_hip_angle": right_hip_angle,
                        "trunk_inclination": trunk_inclination
                    },
                    "label": label
                })

        cap.release()

        output_filename = os.path.splitext(video_file)[0] + '.json'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[✓] Saved {output_path}")
