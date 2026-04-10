"""
Data Preprocessing Module
=========================
Handles data loading, cleaning, normalization, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os


class DataPreprocessor:
    """
    Preprocesses cloud resource data for machine learning models.
    """
    
    def __init__(self, scaler_type='minmax'):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        scaler_type : str
            'minmax' for MinMaxScaler (0-1 range) or 'standard' for StandardScaler
        """
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        self.feature_scaler = None
        self.target_scaler = None
        self.is_fitted = False
    
    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
        
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} records from {file_path}")
        return df
    
    def handle_missing_values(self, df, method='forward_fill'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        method : str
            'forward_fill', 'backward_fill', 'mean', or 'drop'
        
        Returns:
        --------
        pd.DataFrame : Cleaned dataframe
        """
        original_len = len(df)
        
        if method == 'drop':
            df = df.dropna()
        elif method == 'forward_fill':
            df = df.ffill()  # Forward fill (newer pandas syntax)
        elif method == 'backward_fill':
            df = df.bfill()  # Backward fill (newer pandas syntax)
        elif method == 'mean':
            df = df.fillna(df.mean())
        
        df = df.bfill()  # Fill any remaining NaN
        
        if len(df) < original_len:
            print(f"⚠️  Removed {original_len - len(df)} rows with missing values")
        else:
            print("✅ No missing values found")
        
        return df
    
    def create_sequences(self, data, sequence_length=24, prediction_horizon=1):
        """
        Create time-series sequences for LSTM/RNN models.
        
        Parameters:
        -----------
        data : np.array
            Time-series data (features)
        sequence_length : int
            Number of time steps to look back
        prediction_horizon : int
            Number of steps ahead to predict
        
        Returns:
        --------
        tuple : (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence (past values)
            X.append(data[i:i + sequence_length])
            # Target (future values)
            y.append(data[i + sequence_length:i + sequence_length + prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def prepare_lstm_data(self, df, sequence_length=24, test_size=0.2, 
                          features=['cpu_usage', 'ram_usage', 'requests'],
                          targets=['cpu_usage', 'ram_usage']):
        """
        Prepare data specifically for LSTM model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        sequence_length : int
            Number of time steps to look back
        test_size : float
            Proportion of data for testing
        features : list
            Feature columns to use
        targets : list
            Target columns to predict
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, scalers)
        """
        # Extract features and targets
        feature_data = df[features].values
        target_data = df[targets].values
        
        # Scale features
        self.feature_scaler = MinMaxScaler()
        feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
        
        # Scale targets
        self.target_scaler = MinMaxScaler()
        target_data_scaled = self.target_scaler.fit_transform(target_data)
        
        # Create sequences for features (X) - use all features
        X, _ = self.create_sequences(feature_data_scaled, sequence_length)
        
        # Create sequences for targets (y) - use only target columns
        _, y = self.create_sequences(target_data_scaled, sequence_length)
        
        # Reshape y to remove the extra dimension if needed
        # y shape is (samples, 1, n_targets), we need (samples, n_targets)
        if len(y.shape) == 3 and y.shape[1] == 1:
            y = y.reshape(y.shape[0], y.shape[2])
        
        # Split into train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.is_fitted = True
        
        print(f"✅ Prepared LSTM data:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        print(f"   - Sequence length: {sequence_length}")
        print(f"   - Features: {features}")
        print(f"   - Targets: {targets}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_rf_data(self, df, test_size=0.2, lookback=24,
                       features=['cpu_usage', 'ram_usage', 'requests'],
                       targets=['cpu_usage', 'ram_usage']):
        """
        Prepare data for Random Forest model (with lag features).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        test_size : float
            Proportion of data for testing
        lookback : int
            Number of lag features to create
        features : list
            Feature columns to use
        targets : list
            Target columns to predict
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, scaler)
        """
        # Create lag features
        df_processed = df.copy()
        
        for col in features:
            for lag in range(1, lookback + 1):
                df_processed[f'{col}_lag_{lag}'] = df_processed[col].shift(lag)
        
        # Drop rows with NaN (from lag creation)
        df_processed = df_processed.dropna()
        
        # Prepare features (all lag features)
        feature_cols = [f'{col}_lag_{lag}' for col in features for lag in range(1, lookback + 1)]
        X = df_processed[feature_cols].values
        y = df_processed[targets].values
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Scale targets
        self.target_scaler = MinMaxScaler()
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, shuffle=False
        )
        
        self.is_fitted = True
        
        print(f"✅ Prepared Random Forest data:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        print(f"   - Lookback window: {lookback}")
        print(f"   - Total features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_targets(self, y_scaled):
        """
        Inverse transform scaled predictions back to original scale.
        
        Parameters:
        -----------
        y_scaled : np.array
            Scaled predictions
        
        Returns:
        --------
        np.array : Predictions in original scale
        """
        if not self.is_fitted or self.target_scaler is None:
            raise ValueError("Scaler not fitted. Call prepare_*_data first.")
        
        return self.target_scaler.inverse_transform(y_scaled)


if __name__ == "__main__":
    # Test preprocessing
    print("🔄 Testing data preprocessing...")
    
    preprocessor = DataPreprocessor()
    
    # Load data (assuming it exists)
    try:
        df = preprocessor.load_data("data/simulated_cloud_data.csv")
        df = preprocessor.handle_missing_values(df)
        
        # Test LSTM preparation
        print("\n--- LSTM Data Preparation ---")
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = \
            preprocessor.prepare_lstm_data(df, sequence_length=24)
        
        # Test RF preparation
        print("\n--- Random Forest Data Preparation ---")
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = \
            preprocessor.prepare_rf_data(df, lookback=24)
        
        print("\n✅ Preprocessing test completed successfully!")
        
    except FileNotFoundError:
        print("⚠️  Data file not found. Run data_simulation.py first.")

