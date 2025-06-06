import json
import os
import pandas as pd
from pathlib import Path
import re

# Define the set of valid actions
KNOWN_ACTIONS = ['sit_down', 'stand_up', 'still', 'spinning', 'backwards', 'forwards']

def extract_label_from_filename(filename):
    filename = filename.lower()
    for action in KNOWN_ACTIONS:
        if action in filename:
            return action
    return 'unknown'



def convert_landmarks_to_csv():
    # Path to labeled landmarks directory
    landmarks_dir = Path('labeled_landmarks')
    all_data = []
    
    # Read all JSON files
    for json_file in landmarks_dir.glob('*.json'):
        # Extract action label from filename (e.g., 'backwards1.json' -> 'backwards')
        action_label = extract_label_from_filename(json_file.stem)
        print(json_file.stem, action_label)
        
        # Read JSON file
        with open(json_file) as f:
            data = json.load(f)
            
        # Process each frame
        for frame_data in data:
            landmarks = frame_data['landmarks']
            features = frame_data.get('features', {}) # Use .get for backwards compatibility
            
            # Flatten the landmark dictionary
            flat_data = {
                'frame': frame_data['frame'],
                'timestamp': frame_data['timestamp'],
                'action': action_label
            }
            
            # Add each landmark coordinate to the flat dictionary
            for part, coords in landmarks.items():
                flat_data[f'{part}_x'] = coords[0]
                flat_data[f'{part}_y'] = coords[1]
                flat_data[f'{part}_z'] = coords[2]

            # Add the new features
            for feature_name, value in features.items():
                flat_data[feature_name] = value
            
            all_data.append(flat_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_path = 'movement_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print("\nFeatures available:")
    print(df.columns.tolist())

if __name__ == "__main__":
    convert_landmarks_to_csv()