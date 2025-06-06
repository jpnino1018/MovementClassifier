# main.py — Enhanced Real-Time Movement Classifier with Delta Features

import cv2
import mediapipe as mp
import pandas as pd
import joblib
from collections import deque, Counter
import time
import numpy as np

# === Setup ===
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Load the trained model and scaler
model = joblib.load('movement_classifier.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
feature_names = joblib.load('feature_names.joblib')

# Prediction smoothing
pred_buffer = deque(maxlen=10)
last_frame = None

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

def extract_landmark_features(lm):
    features = {}
    features['head_x'], features['head_y'], features['head_z'] = get_xyz(lm, mp_pose.PoseLandmark.NOSE)
    features['shoulders_x'], features['shoulders_y'], features['shoulders_z'] = avg_xyz(lm, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    features['wrists_x'], features['wrists_y'], features['wrists_z'] = avg_xyz(lm, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST)
    features['hips_x'], features['hips_y'], features['hips_z'] = avg_xyz(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
    features['knees_x'], features['knees_y'], features['knees_z'] = avg_xyz(lm, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE)
    features['ankles_x'], features['ankles_y'], features['ankles_z'] = avg_xyz(lm, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE)
    return features

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks or results.pose_landmarks.landmark[0].visibility < 0.6:
        continue

    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    lm = results.pose_landmarks.landmark
    
    # --- Feature Extraction ---
    features = extract_landmark_features(lm)

    # Extract landmarks for angle calculation
    # Left side
    left_shoulder = get_xyz(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
    left_hip = get_xyz(lm, mp_pose.PoseLandmark.LEFT_HIP)
    left_knee = get_xyz(lm, mp_pose.PoseLandmark.LEFT_KNEE)
    left_ankle = get_xyz(lm, mp_pose.PoseLandmark.LEFT_ANKLE)

    # Right side
    right_shoulder = get_xyz(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    right_hip = get_xyz(lm, mp_pose.PoseLandmark.RIGHT_HIP)
    right_knee = get_xyz(lm, mp_pose.PoseLandmark.RIGHT_KNEE)
    right_ankle = get_xyz(lm, mp_pose.PoseLandmark.RIGHT_ANKLE)

    # Midpoints for trunk
    mid_shoulder = features['shoulders_x'], features['shoulders_y'], features['shoulders_z']
    mid_hip = features['hips_x'], features['hips_y'], features['hips_z']

    # Calculate angles
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    vertical_ref_point = [mid_hip[0], mid_hip[1] - 1] 
    trunk_inclination = calculate_angle(vertical_ref_point, mid_hip, mid_shoulder)

    # Add new features to dictionary
    features['left_knee_angle'] = left_knee_angle
    features['left_hip_angle'] = left_hip_angle
    features['right_knee_angle'] = right_knee_angle
    features['right_hip_angle'] = right_hip_angle
    features['trunk_inclination'] = trunk_inclination

    df = pd.DataFrame([features])

    # Add delta features
    if last_frame is not None:
        for col in features:
            df[f'delta_{col}'] = df[col] - last_frame[col]
    else:
        for col in features:
            df[f'delta_{col}'] = 0

    last_frame = features.copy()
    
    # Ensure column order matches the training order
    df = df[feature_names]
    
    scaled = scaler.transform(df)

    probs = model.predict_proba(scaled)[0]
    prediction_encoded = model.predict(scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = max(probs)

    # Smooth predictions
    pred_buffer.append(prediction)
    smoothed_pred = Counter(pred_buffer).most_common(1)[0][0]

    # Display prediction
    cv2.putText(frame, f'Movement: {smoothed_pred} ({confidence:.2f})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display new features
    # Use averages for display to simplify
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
    avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
    cv2.putText(frame, f'Knee Angle: {avg_knee_angle:.1f}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
    cv2.putText(frame, f'Hip Angle: {avg_hip_angle:.1f}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
    cv2.putText(frame, f'Trunk Incl: {trunk_inclination:.1f}', (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

    # Display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow('Movement Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
