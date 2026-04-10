"""
Prediction Module
=================
Makes workload predictions using trained models.
"""

import numpy as np
import pandas as pd
from model_training import LSTMModel, RandomForestModel
from preprocessing import DataPreprocessor
import os


class WorkloadPredictor:
    """
    Predicts future cloud resource usage using trained ML models.
    """
    
    def __init__(self, model_type='lstm', model_path=None):
        """
        Initialize predictor.
        
        Parameters:
        -----------
        model_type : str
            'lstm' or 'rf' (Random Forest)
        model_path : str
            Path to saved model file
        """
        self.model_type = model_type.lower()
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to model file
        """
        if self.model_type == 'lstm':
            self.model = LSTMModel()
            self.model.load(model_path)
        elif self.model_type == 'rf':
            self.model = RandomForestModel()
            self.model.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"✅ Loaded {self.model_type.upper()} model from {model_path}")
    
    def predict_next_step(self, recent_data, preprocessor):
        """
        Predict next time step resource usage.
        
        Parameters:
        -----------
        recent_data : np.array or pd.DataFrame
            Recent historical data (last N time steps)
        preprocessor : DataPreprocessor
            Preprocessor with fitted scalers
        
        Returns:
        --------
        dict : Predictions with keys 'cpu_usage', 'ram_usage'
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare data based on model type
        if self.model_type == 'lstm':
            # For LSTM, need sequence of length sequence_length
            if isinstance(recent_data, pd.DataFrame):
                features = recent_data[['cpu_usage', 'ram_usage', 'requests']].values
            else:
                features = recent_data
            
            # Scale features
            features_scaled = preprocessor.feature_scaler.transform(features)
            
            # Reshape for LSTM: (1, sequence_length, n_features)
            sequence_length = self.model.sequence_length
            if len(features_scaled) < sequence_length:
                raise ValueError(f"Need at least {sequence_length} time steps for LSTM prediction")
            
            # Take last sequence_length steps
            X = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
            
            # Predict
            prediction_scaled = self.model.predict(X)
            
            # Inverse transform
            prediction = preprocessor.target_scaler.inverse_transform(prediction_scaled)[0]
        
        elif self.model_type == 'rf':
            # For RF, need lag features
            if isinstance(recent_data, pd.DataFrame):
                df = recent_data.copy()
            else:
                # Convert array to DataFrame (assuming columns: cpu, ram, requests)
                df = pd.DataFrame(recent_data, columns=['cpu_usage', 'ram_usage', 'requests'])
            
            # Create lag features (assuming lookback=24)
            lookback = 24
            features = ['cpu_usage', 'ram_usage', 'requests']
            
            for col in features:
                for lag in range(1, lookback + 1):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Get last row (most recent lags)
            last_row = df.iloc[-1:][[f'{col}_lag_{lag}' for col in features for lag in range(1, lookback + 1)]]
            
            # Check for NaN
            if last_row.isna().any().any():
                raise ValueError("Insufficient data for lag features. Need at least 24 time steps.")
            
            # Scale and predict
            X = preprocessor.feature_scaler.transform(last_row.values)
            prediction_scaled = self.model.predict(X)
            
            # Inverse transform
            prediction = preprocessor.target_scaler.inverse_transform(prediction_scaled)[0]
        
        return {
            'cpu_usage': max(0, min(100, prediction[0])),  # Clamp to 0-100
            'ram_usage': max(0, min(100, prediction[1])),  # Clamp to 0-100
        }
    
    def predict_multiple_steps(self, historical_data, preprocessor, n_steps=10):
        """
        Predict multiple future time steps (using recursive prediction).
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical data
        preprocessor : DataPreprocessor
            Preprocessor with fitted scalers
        n_steps : int
            Number of steps ahead to predict
        
        Returns:
        --------
        pd.DataFrame : Predictions for next n_steps
        """
        predictions = []
        current_data = historical_data.copy()
        
        for step in range(n_steps):
            # Predict next step
            pred = self.predict_next_step(current_data, preprocessor)
            predictions.append(pred)
            
            # Append prediction to current_data for next iteration
            # (For recursive prediction, we use predicted values)
            last_timestamp = current_data['timestamp'].iloc[-1]
            next_timestamp = last_timestamp + pd.Timedelta(minutes=15)  # Assuming 15-min intervals
            
            # Estimate requests based on CPU (simple heuristic)
            estimated_requests = int((pred['cpu_usage'] / 10) * np.random.uniform(50, 150))
            
            new_row = pd.DataFrame({
                'timestamp': [next_timestamp],
                'cpu_usage': [pred['cpu_usage']],
                'ram_usage': [pred['ram_usage']],
                'requests': [estimated_requests]
            })
            
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return pd.DataFrame(predictions)


if __name__ == "__main__":
    # Test prediction
    print("🔄 Testing prediction module...")
    
    from preprocessing import DataPreprocessor
    from model_training import LSTMModel
    
    try:
        # Load data
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data("data/simulated_cloud_data.csv")
        df = preprocessor.handle_missing_values(df)
        
        # Prepare data (to fit scalers)
        X_train, X_test, y_train, y_test = preprocessor.prepare_lstm_data(df)
        
        # Use recent data for prediction
        recent_data = df.tail(30)  # Last 30 time steps
        
        # Initialize predictor (assuming model exists)
        predictor = WorkloadPredictor(model_type='lstm')
        
        # Note: In real usage, model should be trained first
        print("⚠️  Note: Model needs to be trained before making predictions.")
        print("✅ Prediction module test completed!")
        
    except FileNotFoundError:
        print("⚠️  Data file not found. Run data_simulation.py first.")




