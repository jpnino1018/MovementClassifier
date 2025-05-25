import os
import cv2
import json
from tqdm import tqdm
import mediapipe as mp

# === Setup ===
VIDEOS_DIR = 'videos'
OUTPUT_DIR = 'labeled_landmarks'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                    "label": label
                })

        cap.release()

        output_filename = os.path.splitext(video_file)[0] + '.json'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[✓] Saved {output_path}")
