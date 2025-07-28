# ML--1_PROJECT
#In the ML projects the basic think we need is clean data sheet so the project is on uploading a unclean file we get a clean file which is newly created  

import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import platform
import subprocess

# === Data Cleaning Functions ===

def load_dataset(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            imputer = SimpleImputer(strategy='mean')
        else:
            imputer = SimpleImputer(strategy='most_frequent')
        df[column] = imputer.fit_transform(df[[column]]).ravel()
    return df

def encode_categorical(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

def remove_outliers(df):
    num_df = df.select_dtypes(include=[np.number])
    model = IsolationForest(contamination=0.05, random_state=42)
    inliers = model.fit_predict(num_df)
    return df[inliers == 1]

def normalize_data(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

# === GUI Table Display Function ===

def show_csv_in_gui(csv_file):
    if not os.path.exists(csv_file):
        print(f" File '{csv_file}' not found.")
        return

    df = pd.read_csv(csv_file)

    # Create window
    
    root = tk.Tk()
    root.title(f" Viewing: {csv_file}")

    frame = ttk.Frame(root)
    frame.pack(fill='both', expand=True)

    tree = ttk.Treeview(frame)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # Add scrollbars
    scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    scrollbar_y.pack(side='right', fill='y')
    scrollbar_x.pack(side='bottom', fill='x')
    tree.pack(fill='both', expand=True)

    root.mainloop()

# === Main Cleaning Pipeline ===

def clean_dataset(file_path):
    print(" Loading data...")
    df = load_dataset(file_path)

    print(" Handling missing values...")
    df = handle_missing_values(df)

    # Save Human-Readable Version
    human_file = "cleaned_human_readable.csv"
    print(f" Saving human-readable cleaned data to '{human_file}'...")
    df.to_csv(human_file, index=False)

    # Prepare ML-Ready version
    print(" Preparing ML-ready version...")
    df_ml = df.copy()

    print(" Encoding categorical columns...")
    df_ml, _ = encode_categorical(df_ml)

    print(" Removing outliers...")
    df_ml = remove_outliers(df_ml)

    print(" Normalizing numeric data...")
    df_ml = normalize_data(df_ml)

    ml_file = "cleaned_ml_ready.csv"
    print(f" Saving ML-ready cleaned data to '{ml_file}'...")
    df_ml.to_csv(ml_file, index=False)

    print("\n Cleaning complete.")

    # Show GUI
    print(" Launching GUI to preview cleaned data...")
    show_csv_in_gui(human_file)

# === Main Program ===

if __name__ == "__main__":
    input_path = input(" Enter the path to your dataset (CSV or Excel): ").strip()

    if not os.path.exists(input_path):
        print(" File not found. Please check the path and try again.")
    else:
        clean_dataset(input_path)
