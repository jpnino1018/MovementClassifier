# model.py — Enhanced Version

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
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
    plt.figure(figsize=(12, 8))
    plt.title("Importancia de Características (Feature Importance)")
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera (True Label)')
    plt.xlabel('Etiqueta Predicha (Predicted Label)')
    plt.tight_layout()

def plot_learning_curves(estimator, title, X, y, cv=5, n_jobs=-1):
    from sklearn.model_selection import learning_curve
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Ejemplos de Entrenamiento (Training examples)")
    plt.ylabel("Puntuación (Score)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Puntuación de Entrenamiento (Training score)")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Puntuación de Validación Cruzada (Cross-validation score)")
    
    plt.legend(loc="best")
    plt.tight_layout()

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

    # Encode text labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Balance dataset
    X_balanced, y_balanced = balance_classes(X, y_encoded)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    return X_scaled, y_balanced, scaler, full_feature_cols, label_encoder

def train_model():
    X, y, scaler, feature_names, label_encoder = prepare_data()

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Experimentation ---
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }

    params = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None]
        },
        'XGBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }

    best_model_name = ''
    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"--- Tuning {name} ---")
        grid_search = GridSearchCV(model, params[name], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        
        # Evaluate on test set with the best estimator from the search
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        score = best_estimator.score(X_test, y_test)
        
        print(f"\nModel Performance for {name}:")
        print(f"Accuracy: {score:.4f}")
        print(classification_report(y_test, y_pred))

        if score > best_score:
            best_score = score
            best_model = best_estimator
            best_model_name = name

    print(f"\n--- Best Model Found: {best_model_name} with accuracy {best_score:.4f} ---")

    # --- Generate Plots for the Best Model ---
    final_y_pred = best_model.predict(X_test)

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, final_y_pred, classes=label_encoder.classes_)

    # Plot Feature Importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, feature_names)

    # Plot Learning Curves
    plot_learning_curves(best_model, f"Curvas de Aprendizaje para {best_model_name}", X, y)

    # Save the best model, the scaler, the label encoder, and the feature names
    joblib.dump(best_model, 'movement_classifier.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    print("\nBest model, scaler, label encoder, and feature names saved successfully!")
    
    # Show all plots at the end
    plt.show()

if __name__ == "__main__":
    train_model()
