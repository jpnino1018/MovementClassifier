# model.py — Enhanced Version

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

def add_delta_features(df, landmark_cols):
    delta_df = df.copy()
    for col in landmark_cols:
        delta_df[f'delta_{col}'] = df[col].diff().fillna(0)
    return delta_df

def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()
    plt.show()

def prepare_data(csv_path='movement_dataset.csv'):
    df = pd.read_csv(csv_path)

    # Landmark features only
    landmark_cols = [col for col in df.columns if col not in ['frame', 'timestamp', 'action']]

    # Add delta features
    df = add_delta_features(df, landmark_cols)

    # Recompute feature list
    full_feature_cols = [col for col in df.columns if col not in ['frame', 'timestamp', 'action']]
    X = df[full_feature_cols]
    y = df['action']

    # Balance dataset
    X_balanced, y_balanced = balance_classes(X, y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    return X_scaled, y_balanced, scaler, full_feature_cols

def train_model():
    X, y, scaler, feature_names = prepare_data()

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Feature importance
    plot_feature_importance(model, feature_names)

    # Save
    joblib.dump(model, 'movement_classifier.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("\nModel and scaler saved successfully!")

if __name__ == "__main__":
    train_model()
