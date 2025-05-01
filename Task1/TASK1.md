# 📌 Human Movement Classification Project

## 1. Problem Definition & Methodology

### 🧠 Problem Type
This project addresses a **human activity recognition (HAR)** problem. Specifically, it's a classification task where the goal is to detect and categorize basic human movements (e.g., standing, sitting, walking in, walking out) in real time using video input.

### 🛠️ Methodology Overview
1. **Data Collection**
   - Record videos of people performing the target actions.
   - Build a centralized and shareable video database.

2. **Labeling & Annotation**
   - Use **Label Studio** to annotate action sequences in videos.
   - Tags include temporal segments and action types.

3. **Pose Estimation**
   - Use **MediaPipe** to extract pose landmarks from videos.
   - Focus on landmarks: **hips, knees, ankles, wrists, shoulders, and head**.

4. **Data Preprocessing**
   - Store pose data extracted with MediaPipe in **JSON** format.
   - Match pose data with Label Studio annotations for supervised learning.

5. **Model Training & Evaluation**
   - Use the labeled pose sequences to train a classification model (e.g., RNN, LSTM, Transformer).
   - Evaluate using appropriate metrics.

---

## 2. Pose Data Storage Format

### 🌟 Format: **JSON**
Each video frame with MediaPipe landmarks will be stored as a structured JSON object:
```json
{
  "frame": 24,
  "timestamp": 0.96,
  "landmarks": {
    "head": [x, y, z],
    "shoulders": [...],
    "wrists": [...],
    "hips": [...],
    "knees": [...],
    "ankles": [...]
  },
  "label": "walking"
}
```
Multiple frames make up a sequence corresponding to a labeled action.

### 📚 Storage Strategy:
Pose data (JSON files) will be saved in a shared cloud directory alongside video files for easy access by all collaborators.

#### 📁 Folder Structure Example:
```
/project-root/
├── videos/
│   ├── raw/
│   ├── processed/
│   └── labeled/
├── pose-data/
│   ├── json/
│   │   ├── seq001.json
│   │   ├── seq002.json
│   └── metadata/
│       └── index.csv or json_index.json
├── annotations/ (Label Studio exports)
├── model-training/
│   └── datasets/
```

#### 🔧 Metadata Index File:
Use a `json_index.csv` or `json_index.json` to associate JSON files with their video source and labels:
```
filename,json_path,label,start_time,end_time
seq001.mp4,pose-data/json/seq001.json,walking,0.0,3.5
```

---

## 3. Collaborative Video Database & Infrastructure

### 🌐 Shared Cloud Storage + Metadata Collaboration

Since the project involves multiple collaborators, the recommended setup includes:

- **Video Storage:**
  - Use **Google Drive** (or Dropbox, OneDrive) to store all raw and processed videos.
  - Organize folders: `raw/`, `processed/`, `labeled/`
  - Share the folder among team members with edit access.

- **Metadata Management:**
  - Use a **Google Sheet** to track metadata:
    - Columns: `filename`, `action_label`, `start_time`, `end_time`, `reviewed_by`, `pose_extracted`
    - Allows real-time editing and collaboration

- **Annotation Workflow with Label Studio:**
  - Load videos from Google Drive using shareable links or mount via desktop sync / `rclone`
  - Annotate actions with temporal tags

- **Pose Extraction Workflow:**
  - Team members run MediaPipe scripts locally
  - Extracted pose data is saved in JSON format in a shared or synced folder (`pose-data/json/`)

### 🧹 Benefits:
- Low setup overhead
- Real-time team collaboration
- Easy integration with labeling tools

---

*Next steps may include schema definitions, model training scripts, and evaluation protocols.*

