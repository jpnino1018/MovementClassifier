# MovementClassifier

## Directory Structure

- `videos/`  
  Place your raw video files here, organized in subfolders by action label:
    - sit_down/
    - stand_up/
    - spinning/
    - forwards/
    - backwards/

- `labeled_landmarks/`  
  Output directory for JSON files containing extracted and labeled pose landmarks for each video.

- `src/tools/labeler.py`  
  Script to extract pose landmarks from videos and save them as labeled JSON files.

- `src/tools/grouper.py`  
  Script to combine all labeled JSON files into a single CSV dataset for model training.

- `src/classifier/model.py`  
  Script to train the classifier model using the CSV dataset.

- `src/main.py`  
  Main application for real-time movement classification using your webcam.

## Application Flow

1. **Upload Videos**  
   - Place your labeled videos in the **/videos** directory, each action in its own subfolder (videos mus be previously cut and state the action on the name. **Example: "sit_down1.mp4"**).

2. **Run the Labeler**  
   - Extract pose landmarks from videos:
     ```bash
     python src/tools/labeler.py
     ```

3. **Run the Grouper**  
   - Combine all JSON landmark files into a single CSV:
     ```bash
     python src/tools/grouper.py
     ```

4. **Train the Model**  
   - Train the classifier using the generated CSV:
     ```bash
     python src/classifier/model.py
     ```

5. **Run the Real-Time Classifier**  
   - Start the webcam-based movement classifier:
     ```bash
     python src/main.py
     ```

## Setting Up Python Environment
1. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   ```
2. **Activate**
    - Windows:
    ```cmd
    venv\Scripts\activate
    ```
    - macOS/Linux:
    ```bash
    source venv/bin/activate
    ```
    
3. **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```
    
