import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import argparse
from pathlib import Path

class DataPreprocessor:
    def __init__(self, raw_data_path, output_path='dataset_preprocessing'):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.scaler = None
        self.label_encoder = None
        self.imputer = None
        
        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load raw dataset"""
        print(f"Loading data from {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Perform EDA and return insights"""
        print("\n=== Data Exploration ===")
        print(f"Dataset Info:")
        print(self.df.info())
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if any(missing > 0) else "No missing values")
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        # Save exploration results
        with open(f'{self.output_path}/eda_report.txt', 'w') as f:
            f.write("=== DATA EXPLORATION REPORT ===\n")
            f.write(f"Dataset shape: {self.df.shape}\n")
            f.write(f"Columns: {list(self.df.columns)}\n")
            f.write(f"\nMissing values:\n{self.df.isnull().sum()}\n")
            f.write(f"\nStatistical summary:\n{self.df.describe()}\n")
            
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict()
        }
    
    def preprocess(self, target_column=None):
        """Automated preprocessing pipeline"""
        print("\n=== Starting Preprocessing ===")
        
        # Separate features and target
        if target_column and target_column in self.df.columns:
            X = self.df.drop(target_column, axis=1)
            y = self.df[target_column]
            self.has_target = True
        else:
            X = self.df
            y = None
            self.has_target = False
        
        # Handle missing values
        print("Handling missing values...")
        self.imputer = SimpleImputer(strategy='mean')
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X[numeric_cols]),
                columns=numeric_cols
            )
            # Re-add non-numeric columns if any
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                X_imputed = pd.concat([X_imputed, X[non_numeric_cols].reset_index(drop=True)], axis=1)
        else:
            X_imputed = X.copy()
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        categorical_cols = X_imputed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_imputed[col] = le.fit_transform(X_imputed[col].astype(str))
            # Save encoder for inference
            joblib.dump(le, f'{self.output_path}/encoder_{col}.pkl')
        
        # Scale numerical features
        print("Scaling numerical features...")
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X_imputed.columns
        )
        
        # Handle target variable
        if self.has_target and y is not None:
            if y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = y.values if isinstance(y, pd.Series) else y
                
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, 
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
            
            # Save train test split
            pd.DataFrame(X_train).to_csv(f'{self.output_path}/X_train.csv', index=False)
            pd.DataFrame(X_test).to_csv(f'{self.output_path}/X_test.csv', index=False)
            pd.DataFrame(y_train).to_csv(f'{self.output_path}/y_train.csv', index=False)
            pd.DataFrame(y_test).to_csv(f'{self.output_path}/y_test.csv', index=False)
            
            print(f"Training set: {X_train.shape}")
            print(f"Test set: {X_test.shape}")
            
        else:
            # Save full preprocessed dataset
            X_scaled.to_csv(f'{self.output_path}/preprocessed_data.csv', index=False)
            print(f"Preprocessed data shape: {X_scaled.shape}")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, f'{self.output_path}/scaler.pkl')
        joblib.dump(self.imputer, f'{self.output_path}/imputer.pkl')
        if self.label_encoder:
            joblib.dump(self.label_encoder, f'{self.output_path}/target_encoder.pkl')
        
        # Save preprocessing metadata
        metadata = {
            'original_shape': self.df.shape,
            'preprocessed_shape': X_scaled.shape,
            'numeric_columns': list(numeric_cols),
            'categorical_columns': list(categorical_cols),
            'has_target': self.has_target,
            'target_column': target_column if self.has_target else None
        }
        
        import json
        with open(f'{self.output_path}/preprocessing_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\nPreprocessing completed successfully!")
        return X_scaled, y_encoded if self.has_target else None
    
    def run_pipeline(self, target_column=None):
        """Run complete preprocessing pipeline"""
        self.load_data()
        self.explore_data()
        return self.preprocess(target_column)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated Data Preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Path to raw dataset')
    parser.add_argument('--output', type=str, default='dataset_preprocessing', help='Output directory')
    parser.add_argument('--target', type=str, default=None, help='Target column name')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocessor = DataPreprocessor(args.input, args.output)
    X, y = preprocessor.run_pipeline(target_column=args.target)
    
    print(f"\nPreprocessed data saved to: {args.output}")