
import pandas as pd
import numpy as np
import os
import json

def preprocess_data(file_path):
    """
    Preprocess the dataset and return the cleaned DataFrame along with a preprocessing receipt.
    """
    # Read the dataset
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    except Exception as e:
        return None, {"error": str(e)}

    receipt = {"file": os.path.basename(file_path), "operations": []}

    # Record initial dataset information
    receipt["initial_shape"] = df.shape
    receipt["initial_missing_values"] = df.isnull().sum().to_dict()
    receipt["initial_column_types"] = df.dtypes.astype(str).to_dict()

    # Attempt to convert columns that may represent dates
    for col in df.columns:
        if df[col].dtype == "object":  # Only consider object (string) columns
            try:
                # Try converting string columns to datetime
                df[col] = pd.to_datetime(df[col], errors="raise")
                receipt["operations"].append(f"Column '{col}' successfully converted to datetime.")
            except Exception:
                # If conversion fails, it stays as object (string)
                pass

    # Handle missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    for col in missing_columns:
        if df[col].dtype in ["float64", "int64"]:  # Numeric columns
            df[col].fillna(df[col].mean(), inplace=True)
            receipt["operations"].append(f"Missing values in column '{col}' replaced with mean.")
        elif df[col].dtype == "datetime64[ns]":  # Datetime columns
            receipt["operations"].append(f"Missing values in datetime column '{col}' handled (no imputation).")
        else:  # Non-numeric columns
            df[col].fillna("Unknown", inplace=True)
            receipt["operations"].append(f"Missing values in column '{col}' replaced with 'Unknown'.")

    # Handle categorical data, excluding datetime columns
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_columns:
        if df[col].dtype != "datetime64[ns]":  # Exclude datetime columns
            df[col] = df[col].astype("category").cat.codes
            receipt["operations"].append(f"Categorical column '{col}' encoded as numeric codes.")

    # Scale numeric columns
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    for col in numeric_columns:
        min_val, max_val = df[col].min(), df[col].max()
        if min_val != max_val:  # Avoid division by zero
            df[col] = (df[col] - min_val) / (max_val - min_val)
            receipt["operations"].append(f"Numeric column '{col}' scaled to range 0-1.")

    # Final dataset information
    receipt["final_shape"] = df.shape
    receipt["final_missing_values"] = df.isnull().sum().to_dict()

    return df, receipt


def save_receipt(receipt, output_path):
    """
    Save the preprocessing receipt as a JSON file.
    """
    with open(output_path, "w") as file:
        json.dump(receipt, file, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a dataset.")
    parser.add_argument("file", help="Path to the dataset file (CSV or Excel).")
    parser.add_argument("output", help="Path to save the preprocessed dataset.")
    parser.add_argument(
        "--receipt",
        help="Path to save the preprocessing receipt (default: receipt.json).",
        default="receipt.json",
    )

    args = parser.parse_args()

    # Preprocess the dataset
    preprocessed_df, receipt = preprocess_data(args.file)

    if preprocessed_df is not None:
        # Save the preprocessed dataset
        preprocessed_df.to_csv(args.output, index=False)

        # Save the receipt
        save_receipt(receipt, args.receipt)

        print(f"Preprocessing completed! Results saved to {args.output} and {args.receipt}.")
    else:
        print(f"Preprocessing failed: {receipt['error']}")