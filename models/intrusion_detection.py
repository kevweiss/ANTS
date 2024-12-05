import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import mode
import joblib

# Get the current working directory (repo root)
repo_root = os.path.dirname(os.path.abspath(__file__))

# Define the folder containing the data
data_folder = os.path.join(repo_root, 'data')

# Function to load dataset files dynamically
def load_data(data_folder):
    # List of files to load
    filenames = [
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv'
    ]
    
    # Load all datasets and concatenate them into one DataFrame
    data_frames = []
    for file in filenames:
        file_path = os.path.join(data_folder, file)
        data = pd.read_csv(file_path)
        data_frames.append(data)
    
    # Concatenate all data into one DataFrame
    full_data = pd.concat(data_frames, ignore_index=True)
    
    # Label encoding the 'Label' column (normal vs attack)
    label_encoder = LabelEncoder()
    full_data['Label'] = label_encoder.fit_transform(full_data['Label'])

    # Separate features and labels
    X = full_data.drop('Label', axis=1)
    y = full_data['Label']

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y

# Load data
X, y = load_data(data_folder)

# Train Random Forest Model
def train_rf(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    return rf_clf

# Train XGBoost Model
def train_xgb(X_train, y_train):
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, n_jobs=-1)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

# Train Deep Neural Network Model
def train_dnn(X_train, y_train, input_dim):
    dnn_model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=2)
    return dnn_model

# Evaluate the ensemble model
def evaluate_model(models, X_test, y_test):
    rf_pred = models['rf'].predict(X_test)
    xgb_pred = models['xgb'].predict(X_test)
    dnn_pred = (models['dnn'].predict(X_test) > 0.5).astype(int).flatten()

    # Combine predictions using majority voting
    combined_preds = np.array([rf_pred, xgb_pred, dnn_pred]).T
    ensemble_pred = mode(combined_preds, axis=1).mode.flatten()

    # Evaluate performance
    print("Accuracy:", accuracy_score(y_test, ensemble_pred))
    print("Classification Report:\n", classification_report(y_test, ensemble_pred))

# Save models to disk
def save_models(models):
    joblib.dump(models['rf'], 'rf_model.pkl')
    joblib.dump(models['xgb'], 'xgb_model.pkl')
    models['dnn'].save('dnn_model.h5')

# Main function to execute the steps
def main():
    # Split the data into training and testing (70/30 split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    rf_clf = train_rf(X_train, y_train)
    xgb_clf = train_xgb(X_train, y_train)
    dnn_model = train_dnn(X_train, y_train, X_train.shape[1])

    # Store trained models
    models = {
        'rf': rf_clf,
        'xgb': xgb_clf,
        'dnn': dnn_model
    }

    # Evaluate the model
    evaluate_model(models, X_test, y_test)

    # Save models for later use
    save_models(models)

if __name__ == '__main__':
    main()
