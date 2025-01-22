import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

def load_preprocessed_data(file_path):
    """
    Load preprocessed data from the given file path.
    """
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        print("Preprocessed data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None

def preprocess_features(data, target_column):
    """
    Preprocess features: drop non-numeric columns and convert timestamps.
    """
    # Drop non-numeric columns (e.g., 'trans_date_trans_time', 'cc_num', 'trans_num', etc.)
    non_numeric_columns = ['trans_date_trans_time', 'cc_num', 'trans_num', 'dob', 'street', 'city', 'state', 'zip', 'merchant', 'category', 'first', 'last', 'gender', 'job', 'city_pop']
    data = data.drop(columns=non_numeric_columns)

    # Convert timestamps to numeric (if necessary, for example: number of seconds since reference date)
    if 'trans_date_trans_time' in data.columns:
        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
        data['trans_date_trans_time'] = (data['trans_date_trans_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')  # Convert to seconds

    print("\nData after dropping non-numeric columns and handling timestamps:")
    print(data.head())
    
    # Return features (X) and target (y)
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    return X, y

def train_model(data, target_column):
    """
    Train a Random Forest model on the given data.
    """
    print("Training the model...")

    # Preprocess features and target
    X, y = preprocess_features(data, target_column)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.2f}")

    return model

def save_model(model, output_file):
    """
    Save the trained model to a file.
    Ensure the directory exists.
    """
    try:
        # Ensure the 'models' directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        joblib.dump(model, output_file)
        print(f"Model saved to {output_file}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    # File paths
    input_file = "c:/users/vamshi/full_stack/fraud_detection/data/preprocessed_data_with_labels.xlsx"  # Correct file name after preprocessing
    model_file = "c:/users/vamshi/full_stack/fraud_detection/models/fraud_detection_model.pkl"
    target_column = "is_fraud"  # Update with the correct target column name

    # Load preprocessed data
    data = load_preprocessed_data(input_file)
    if data is not None:
        # Train and save the model
        model = train_model(data, target_column)
        save_model(model, model_file)