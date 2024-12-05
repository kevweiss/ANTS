# Import necessary libraries
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

# Load the NSL-KDD dataset
def load_data(train_file, test_file, columns):
    train_data = pd.read_csv(train_file, names=columns)
    test_data = pd.read_csv(test_file, names=columns)

    # One-hot encode categorical features
    categorical_features = ['protocol_type', 'service', 'flag']
    train_data = pd.get_dummies(train_data, columns=categorical_features)
    test_data = pd.get_dummies(test_data, columns=categorical_features)

    # Align test dataset to training columns
    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

    # Encode labels
    label_encoder = LabelEncoder()
    train_data['label'] = label_encoder.fit_transform(train_data['label'])
    test_data['label'] = label_encoder.transform(test_data['label'])

    # Separate features and labels
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    # Normalize features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

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
    # Define column names for the NSL-KDD dataset
    columns = [...]  # Fill with the actual columns based on the dataset documentation
    
    # Load data
    X_train, y_train, X_test, y_test = load_data('KDDTrain+.csv', 'KDDTest+.csv', columns)
    
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
