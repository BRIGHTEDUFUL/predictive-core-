import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os

def load_and_preprocess_data():
    """
    Load the student performance dataset. 
    If student_data.csv exists, it loads it. Otherwise, it generates synthetic data.
    """
    data_path = 'student_data.csv'
    features = ['study_time', 'failures', 'absences', 'G1', 'G2']
    
    if os.path.exists(data_path):
        print(f"[*] Loading professional training dataset from {data_path}...")
        data = pd.read_csv(data_path)
        # Basic validation: check if required columns exist
        required_cols = features + ['pass']
        if not all(col in data.columns for col in required_cols):
            if 'final_grade' in data.columns and 'pass' not in data.columns:
                data['pass'] = (data['final_grade'] >= 10).astype(int)
            else:
                raise ValueError(f"CSV must contain at least headers: {', '.join(features)} and either 'pass' or 'final_grade'")
    else:
        print("[!] No student_data.csv found. Generating high-fidelity synthetic training data...")
        np.random.seed(42)
        n_samples = 1500
        study_time = np.random.uniform(1, 5, n_samples)
        failures = np.random.poisson(0.5, n_samples)
        absences = np.random.poisson(5, n_samples)
        g1 = np.random.uniform(0, 20, n_samples)
        g2 = np.random.uniform(0, 20, n_samples)
        
        # Complex heuristic for realistic grade distribution
        base_grade = (g1 * 0.3) + (g2 * 0.4) + (study_time * 1.5) - (failures * 2.5) - (absences * 0.15)
        final_grade = np.clip(base_grade + np.random.normal(0, 1.5, n_samples), 0, 20)
        
        data = pd.DataFrame({
            'study_time': study_time,
            'failures': failures,
            'absences': absences,
            'G1': g1,
            'G2': g2,
            'final_grade': final_grade
        })
        data['pass'] = (data['final_grade'] >= 10).astype(int)
        
        # Save synthetic data for user reference as a template
        data.to_csv(data_path, index=False)
        print(f"[+] Synthetic dataset generated and saved to {data_path} (n={n_samples})")
    
    X = data[features]
    y = data['pass']
    
    return X, y

def train_and_evaluate_model():
    """
    Train the Random Forest model with hyperparameter tuning and evaluate performance
    """
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Handle missing values if present
    X = X.fillna(X.mean())
    
    # Split data into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save the model and scaler
    joblib.dump(best_model, 'student_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Create visualizations
    create_visualizations(cm, accuracy, precision, recall, f1)
    
    return best_model, scaler

def create_visualizations(cm, accuracy, precision, recall, f1):
    """
    Create and save visualizations for model evaluation
    """
    # Create confusion matrix plot
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Fail', 'Pass'])
    plt.yticks(tick_marks, ['Fail', 'Pass'])
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Create bar chart for performance metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values = [accuracy, precision, recall, f1]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()

if __name__ == "__main__":
    print("Starting model training process...")
    model, scaler = train_and_evaluate_model()
    print("Model training completed. Files saved:")
    print("- student_model.pkl (trained model)")
    print("- scaler.pkl (feature scaler)")
    print("- confusion_matrix.png (visualization)")
    print("- performance_metrics.png (visualization)")