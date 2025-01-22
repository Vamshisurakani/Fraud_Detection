import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, target_column):
    """
    Perform data preprocessing:
    - Handle missing values
    - Scale the 'Amount' column
    """
    print("Preprocessing data...")

    # Check for missing values
    if data.isnull().sum().any():
        print("Missing values found. Filling with median...")
        data.fillna(data.median(), inplace=True)

    # Scale the 'Amount' column
    if 'Amount' in data.columns:
        scaler = StandardScaler()
        data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
        print("'Amount' column scaled.")

    # Ensure the target column exists
    if target_column not in data.columns:
        print(f"Warning: '{target_column}' column is missing.")
    else:
        print(f"'{target_column}' column found.")

    # Print column names for verification
    print("\nColumns after preprocessing:")
    print(data.columns)

    print("Preprocessing complete!")
    return data

def save_preprocessed_data(data, output_file):
    """
    Save the preprocessed data to an Excel file.
    """
    try:
        data.to_excel(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")

if __name__ == "__main__":
    # File paths
    input_file = "c:/users/vamshi/full_stack/fraud_detection/data/creditcard.xlsx"  # Replace with your actual file path
    output_file = "c:/users/vamshi/full_stack/fraud_detection/data/preprocessed_data_with_labels.xlsx"  # Save as preprocessed data with target

    # Specify the target column (fraud label column)
    target_column = 'Class'  # Change if your dataset has a different target column

    # Load and preprocess data
    raw_data = load_data(input_file)
    if raw_data is not None:
        preprocessed_data = preprocess_data(raw_data, target_column)
        save_preprocessed_data(preprocessed_data, output_file)
