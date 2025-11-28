"""
Model Training with Cross-Validation and Hyperparameter Tuning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import logging
from typing import Dict, Any, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxiModelTrainer:
    """
    Train and evaluate multiple regression models for tip prediction
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets
        """
        logger.info("Splitting data into train/test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            shuffle=True
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_models(self, X_train, y_train, X_test, y_test):
        """
        Train linear regression models with regularization
        """
        logger.info("\n" + "="*60)
        logger.info("Training Linear Models")
        logger.info("="*60)
        
        # 1. Simple Linear Regression
        logger.info("\n1. Linear Regression (OLS)")
        lr = LinearRegression()
        
        start_time = time.time()
        lr.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        train_pred = lr.predict(X_train)
        test_pred = lr.predict(X_test)
        
        self.models['linear_regression'] = lr
        self.results['linear_regression'] = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_time': train_time
        }
        
        self._print_metrics('Linear Regression', self.results['linear_regression'])
        
        # 2. Ridge Regression with hyperparameter tuning
        logger.info("\n2. Ridge Regression (L2 Regularization)")
        
        ridge_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky']
        }
        
        ridge = Ridge(random_state=self.random_state)
        ridge_cv = GridSearchCV(
            ridge, ridge_params, 
            cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        ridge_cv.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        logger.info(f"Best parameters: {ridge_cv.best_params_}")
        
        train_pred = ridge_cv.predict(X_train)
        test_pred = ridge_cv.predict(X_test)
        
        self.models['ridge'] = ridge_cv.best_estimator_
        self.results['ridge'] = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_time': train_time,
            'best_params': ridge_cv.best_params_
        }
        
        self._print_metrics('Ridge Regression', self.results['ridge'])
        
        # 3. Lasso Regression
        logger.info("\n3. Lasso Regression (L1 Regularization)")
        
        lasso_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'max_iter': [1000, 5000]
        }
        
        lasso = Lasso(random_state=self.random_state)
        lasso_cv = GridSearchCV(
            lasso, lasso_params,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        lasso_cv.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        logger.info(f"Best parameters: {lasso_cv.best_params_}")
        
        train_pred = lasso_cv.predict(X_train)
        test_pred = lasso_cv.predict(X_test)
        
        self.models['lasso'] = lasso_cv.best_estimator_
        self.results['lasso'] = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_time': train_time,
            'best_params': lasso_cv.best_params_
        }
        
        self._print_metrics('Lasso Regression', self.results['lasso'])
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest with hyperparameter tuning
        """
        logger.info("\n" + "="*60)
        logger.info("Training Random Forest")
        logger.info("="*60)
        
        # Randomized search for faster tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        }
        
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        rf_cv = RandomizedSearchCV(
            rf, rf_params,
            n_iter=20,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=2,
            random_state=self.random_state
        )
        
        start_time = time.time()
        rf_cv.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        logger.info(f"\nBest parameters: {rf_cv.best_params_}")
        
        train_pred = rf_cv.predict(X_train)
        test_pred = rf_cv.predict(X_test)
        
        self.models['random_forest'] = rf_cv.best_estimator_
        self.results['random_forest'] = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_time': train_time,
            'best_params': rf_cv.best_params_
        }
        
        self._print_metrics('Random Forest', self.results['random_forest'])
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """
        Train a neural network using Keras
        """
        logger.info("\n" + "="*60)
        logger.info("Training Neural Network")
        logger.info("="*60)
        
        # Build model
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train model
        logger.info("Training neural network...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=256,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        # Predictions
        train_pred = model.predict(X_train, verbose=0).flatten()
        test_pred = model.predict(X_test, verbose=0).flatten()
        
        self.models['neural_network'] = model
        self.results['neural_network'] = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_time': train_time,
            'history': history.history
        }
        
        self._print_metrics('Neural Network', self.results['neural_network'])
    
    def _print_metrics(self, model_name: str, metrics: Dict):
        """
        Print model metrics in a formatted way
        """
        print(f"\n{model_name} Results:")
        print(f"  Train MAE:  ${metrics['train_mae']:.2f}")
        print(f"  Test MAE:   ${metrics['test_mae']:.2f}")
        print(f"  Train RMSE: ${metrics['train_rmse']:.2f}")
        print(f"  Test RMSE:  ${metrics['test_rmse']:.2f}")
        print(f"  Train R²:   {metrics['train_r2']:.4f}")
        print(f"  Test R²:    {metrics['test_r2']:.4f}")
        print(f"  Train Time: {metrics['train_time']:.2f}s")
    
    def select_best_model(self):
        """
        Select the best model based on test MAE
        """
        best_mae = float('inf')
        
        for name, metrics in self.results.items():
            if metrics['test_mae'] < best_mae:
                best_mae = metrics['test_mae']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Best Model: {self.best_model_name}")
        logger.info(f"Test MAE: ${best_mae:.2f}")
        logger.info(f"{'='*60}")
        
        return self.best_model, self.best_model_name
    
    def save_models(self, path: str = 'models/'):
        """
        Save all trained models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(f"{path}{name}.keras")
            else:
                joblib.dump(model, f"{path}{name}.pkl")
        
        # Save results
        import json
        results_serializable = {}
        for name, metrics in self.results.items():
            results_serializable[name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v 
                for k, v in metrics.items() 
                if k != 'history'
            }
        
        with open(f"{path}results.json", 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Models saved to {path}")


# Main execution
if __name__ == "__main__":
    # Load preprocessed data
    X = np.load('X_processed.npy')
    y = np.load('y_processed.npy')
    
    print(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    
    # Initialize trainer
    trainer = TaxiModelTrainer(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)
    
    # Train all models
    trainer.train_linear_models(X_train, y_train, X_test, y_test)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    trainer.train_neural_network(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_model, best_name = trainer.select_best_model()
    
    # Save models
    trainer.save_models()
    
    print("\n✓ Model training complete!")