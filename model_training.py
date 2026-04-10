"""
Model Training Module
=====================
Trains LSTM and Random Forest models for cloud resource usage prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle


class LSTMModel:
    """
    LSTM model for time-series prediction of cloud resources.
    """
    
    def __init__(self, sequence_length=24, n_features=3, n_targets=2):
        """
        Initialize LSTM model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        n_features : int
            Number of input features
        n_targets : int
            Number of target variables to predict
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_targets = n_targets
        self.model = None
        self.history = None
    
    def build_model(self, lstm_units=[64, 32], dropout_rate=0.2):
        """
        Build LSTM model architecture.
        
        Parameters:
        -----------
        lstm_units : list
            Number of units in each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True if len(lstm_units) > 1 else False,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for units in lstm_units[1:]:
            model.add(LSTM(units=units, return_sequences=False))
            model.add(Dropout(dropout_rate))
        
        # Dense output layer
        model.add(Dense(self.n_targets))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print("✅ LSTM model built successfully")
        print(model.summary())
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        X_train : np.array
            Training sequences
        y_train : np.array
            Training targets
        X_val : np.array
            Validation sequences
        y_val : np.array
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
        
        Returns:
        --------
        dict : Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/lstm_best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("✅ LSTM model training completed")
        return self.history.history
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.array
            Input sequences
        
        Returns:
        --------
        np.array : Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test, scaler=None):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X_test : np.array
            Test sequences
        y_test : np.array
            Test targets
        scaler : object
            Scaler to inverse transform predictions
        
        Returns:
        --------
        dict : Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        # Inverse transform if scaler provided
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            y_test = scaler.inverse_transform(y_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2
        }
        
        print("\n📊 LSTM Model Evaluation:")
        print(f"   MSE:  {mse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        return metrics
    
    def save(self, filepath='models/lstm_model.h5'):
        """
        Save model to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath='models/lstm_model.h5'):
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        try:
            # Try loading with compile=False to avoid metric deserialization issues
            self.model = tf.keras.models.load_model(filepath, compile=False)
            # Recompile with the same settings
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            print(f"✅ Model loaded from {filepath}")
        except Exception as e:
            print(f"⚠️  Could not load model from {filepath}: {e}")
            print("   The model file may be from an incompatible Keras version.")
            print("   Please delete the old model file and retrain, or the model will be retrained automatically.")
            raise ValueError(f"Model loading failed. Please delete {filepath} and retrain.")


class RandomForestModel:
    """
    Random Forest model for cloud resource prediction.
    """
    
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Initialize Random Forest model.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of trees
        random_state : int
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
    
    def train(self, X_train, y_train):
        """
        Train Random Forest model.
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training targets
        """
        print("🔄 Training Random Forest model...")
        
        # Train separate models for each target
        self.models = []
        for i in range(y_train.shape[1]):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            model.fit(X_train, y_train[:, i])
            self.models.append(model)
        
        self.model = self.models  # For compatibility
        print("✅ Random Forest model training completed")
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.array
            Input features
        
        Returns:
        --------
        np.array : Predictions
        """
        if self.models is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def evaluate(self, X_test, y_test, scaler=None):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X_test : np.array
            Test features
        y_test : np.array
            Test targets
        scaler : object
            Scaler to inverse transform predictions
        
        Returns:
        --------
        dict : Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        # Inverse transform if scaler provided
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            y_test = scaler.inverse_transform(y_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2
        }
        
        print("\n📊 Random Forest Model Evaluation:")
        print(f"   MSE:  {mse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")
        
        return metrics
    
    def save(self, filepath='models/rf_model.pkl'):
        """
        Save model to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if self.models is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath='models/rf_model.pkl'):
        """
        Load model from file.
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        with open(filepath, 'rb') as f:
            self.models = pickle.load(f)
        self.model = self.models
        print(f"✅ Model loaded from {filepath}")


if __name__ == "__main__":
    # Test model training
    print("🔄 Testing model training...")
    
    from preprocessing import DataPreprocessor
    
    # Load and prepare data
    preprocessor = DataPreprocessor()
    try:
        df = preprocessor.load_data("data/simulated_cloud_data.csv")
        df = preprocessor.handle_missing_values(df)
        
        # Prepare LSTM data
        X_train, X_test, y_train, y_test = preprocessor.prepare_lstm_data(df)
        
        # Split train into train and validation
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        # Train LSTM
        print("\n--- Training LSTM Model ---")
        lstm_model = LSTMModel(sequence_length=24, n_features=3, n_targets=2)
        lstm_model.build_model()
        lstm_model.train(X_train, y_train, X_val, y_val, epochs=10, verbose=1)
        lstm_model.evaluate(X_test, y_test, preprocessor.target_scaler)
        
        print("\n✅ Model training test completed!")
        
    except FileNotFoundError:
        print("⚠️  Data file not found. Run data_simulation.py first.")

