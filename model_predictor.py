"""
Reusable ML Model Predictor Module
Use this in your existing web app to make predictions
"""

import joblib
import pandas as pd
import numpy as np
import os

class TimeToFailurePredictor:
    """Wrapper class for the Time-to-Failure prediction model"""
    
    def __init__(self, model_dir='.'):
        """
        Initialize the predictor by loading model and preprocessing objects
        
        Args:
            model_dir: Directory where model files are stored (default: current directory)
        """
        self.model_dir = model_dir
        self.model = None
        self.label_encoders = None
        self.scaler = None
        self.feature_info = None
        self._load_model()
    
    def _load_model(self):
        """Load all required model files"""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'best_model.pkl'))
            self.label_encoders = joblib.load(os.path.join(self.model_dir, 'label_encoders.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.feature_info = joblib.load(os.path.join(self.model_dir, 'feature_info.pkl'))
            print("Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _feature_engineering(self, df):
        """Apply feature engineering (same as training)"""
        # Create interaction features
        if 'equipment_age_days' in df.columns and 'days_since_maintenance' in df.columns:
            df['maintenance_ratio'] = df['days_since_maintenance'] / (df['equipment_age_days'] + 1)
            df['age_maintenance_interaction'] = df['equipment_age_days'] * df['days_since_maintenance']
        
        # Create time-based features
        if 'month' in df.columns:
            df['is_peak_season'] = df['month'].isin([6, 7, 8, 12, 1]).astype(int)
        
        if 'hour_of_day' in df.columns:
            df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        # Create cost efficiency features
        if 'cost_estimate' in df.columns and 'downtime_hours' in df.columns:
            df['cost_per_hour'] = df['cost_estimate'] / (df['downtime_hours'] + 1)
        
        # Facility size features
        if 'total_area_sqm' in df.columns and 'floor_count' in df.columns:
            df['area_per_floor'] = df['total_area_sqm'] / (df['floor_count'] + 1)
        
        # Equipment age categories
        if 'equipment_age_days' in df.columns:
            df['equipment_age_years'] = df['equipment_age_days'] / 365.25
            df['is_old_equipment'] = (df['equipment_age_days'] > 2000).astype(int)
            df['is_new_equipment'] = (df['equipment_age_days'] < 500).astype(int)
        
        # Maintenance frequency features
        if 'maintenance_cycle_days' in df.columns:
            df['maintenance_frequency'] = 365.25 / (df['maintenance_cycle_days'] + 1)
            df['is_high_maintenance'] = (df['maintenance_cycle_days'] < 90).astype(int)
        
        return df
    
    def _preprocess(self, input_data):
        """Preprocess input data (encoding, scaling, etc.)"""
        # Convert to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be dict or pandas DataFrame")
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        # Encode categorical variables
        for col in self.feature_info['categorical_cols']:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le is not None:
                    # Handle unseen categories
                    try:
                        df[col] = le.transform(df[col].astype(str))
                    except ValueError:
                        # If category not seen during training, use most common
                        df[col] = 0
        
        # Handle missing values
        for col in self.feature_info['numerical_cols']:
            if col in df.columns:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col].fillna(median_val, inplace=True)
        
        for col in self.feature_info['categorical_cols']:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(0, inplace=True)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in self.feature_info['numerical_cols']:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Scale numerical features
        if self.feature_info.get('use_scaler', False):
            numerical_cols = [col for col in self.feature_info['numerical_cols'] if col in df.columns]
            if numerical_cols:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Ensure correct feature order
        feature_order = self.feature_info['feature_order']
        missing_cols = set(feature_order) - set(df.columns)
        if missing_cols:
            # Add missing columns with default values
            for col in missing_cols:
                df[col] = 0
        
        df = df[feature_order]
        
        return df
    
    def predict(self, input_data):
        """
        Make prediction for input data
        
        Args:
            input_data: Dictionary or DataFrame with feature values
        
        Returns:
            float: Predicted days to next failure
        """
        try:
            # Preprocess
            processed_data = self._preprocess(input_data)
            
            # Predict
            prediction = self.model.predict(processed_data)[0]
            
            return float(prediction)
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def predict_batch(self, input_list):
        """
        Make predictions for multiple inputs
        
        Args:
            input_list: List of dictionaries or DataFrame
        
        Returns:
            list: List of predicted days to next failure
        """
        try:
            df = pd.DataFrame(input_list)
            processed_data = self._preprocess(df)
            predictions = self.model.predict(processed_data)
            return [float(p) for p in predictions]
        except Exception as e:
            raise Exception(f"Batch prediction error: {str(e)}")


# Example usage function
def get_predictor(model_dir='.'):
    """Helper function to get a predictor instance"""
    return TimeToFailurePredictor(model_dir)

