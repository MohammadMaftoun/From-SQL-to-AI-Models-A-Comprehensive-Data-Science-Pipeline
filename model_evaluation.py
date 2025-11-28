"""
Comprehensive Model Evaluation and Interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
import shap
import joblib
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation and interpretability
    """
    
    def __init__(self, model, model_name: str, feature_names: List[str]):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        
    def evaluate_regression(self, X_test, y_test, y_pred=None):
        """
        Comprehensive regression evaluation
        """
        if y_pred is None:
            y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Regression Evaluation: {self.model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Mean Absolute Error (MAE):  ${mae:.2f}")
        logger.info(f"Root Mean Squared Error:    ${rmse:.2f}")
        logger.info(f"R² Score:                   {r2:.4f}")
        logger.info(f"Mean Absolute % Error:      {mape:.2f}%")
        
        # Residual analysis
        residuals = y_test - y_pred
        logger.info(f"\nResidual Analysis:")
        logger.info(f"  Mean residual:     ${np.mean(residuals):.2f}")
        logger.info(f"  Std residual:      ${np.std(residuals):.2f}")
        logger.info(f"  Min residual:      ${np.min(residuals):.2f}")
        logger.info(f"  Max residual:      ${np.max(residuals):.2f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'residuals': residuals
        }
    
    def plot_predictions(self, y_test, y_pred, save_path='plots/predictions.png'):
        """
        Plot actual vs predicted values
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
        axes[0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('Actual Tip Amount ($)', fontsize=12)
        axes[0].set_ylabel('Predicted Tip Amount ($)', fontsize=12)
        axes[0].set_title(f'{self.model_name}: Actual vs Predicted', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Tip Amount ($)', fontsize=12)
        axes[1].set_ylabel('Residuals ($)', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction plot saved to {save_path}")
        plt.close()
    
    def plot_residual_distribution(self, residuals, save_path='plots/residual_dist.png'):
        """
        Plot residual distribution
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Residuals ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Residual Distribution', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residual distribution plot saved to {save_path}")
        plt.close()
    
    def plot_error_by_range(self, y_test, y_pred, save_path='plots/error_by_range.png'):
        """
        Plot error metrics by tip amount ranges
        """
        # Create bins
        bins = [0, 2, 5, 10, 20, 100]
        labels = ['$0-2', '$2-5', '$5-10', '$10-20', '$20+']
        
        df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'error': np.abs(y_test - y_pred)
        })
        
        df['range'] = pd.cut(df['actual'], bins=bins, labels=labels)
        
        # Calculate metrics per range
        metrics_by_range = df.groupby('range').agg({
            'error': ['mean', 'std', 'count']
        }).reset_index()
        
        metrics_by_range.columns = ['range', 'mean_error', 'std_error', 'count']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Mean error by range
        axes[0].bar(metrics_by_range['range'], metrics_by_range['mean_error'])
        axes[0].set_xlabel('Tip Amount Range', fontsize=12)
        axes[0].set_ylabel('Mean Absolute Error ($)', fontsize=12)
        axes[0].set_title('Error by Tip Amount Range', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Sample count by range
        axes[1].bar(metrics_by_range['range'], metrics_by_range['count'])
        axes[1].set_xlabel('Tip Amount Range', fontsize=12)
        axes[1].set_ylabel('Number of Samples', fontsize=12)
        axes[1].set_title('Sample Distribution', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Error by range plot saved to {save_path}")
        plt.close()
    
    def shap_analysis(self, X_test, sample_size=1000, save_path='plots/shap_'):
        """
        SHAP analysis for model interpretability
        """
        logger.info("\nPerforming SHAP analysis...")
        
        # Sample data for faster computation
        if len(X_test) > sample_size:
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[indices]
        else:
            X_sample = X_test
        
        try:
            # Create explainer
            if self.model_name in ['random_forest', 'gradient_boosting']:
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.KernelExplainer(self.model.predict, X_sample[:100])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=self.feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(f'{save_path}summary.png', dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}summary.png")
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig(f'{save_path}importance.png', dpi=300, bbox_inches='tight')
            logger.info(f"SHAP importance plot saved to {save_path}importance.png")
            plt.close()
            
            # Get feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Most Important Features (SHAP):")
            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return shap_values, importance_df
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            return None, None
    
    def feature_importance_sklearn(self, save_path='plots/feature_importance.png'):
        """
        Plot feature importance for sklearn models
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Top 15 features
            top_n = min(15, len(self.feature_names))
            top_indices = indices[:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(top_n), importances[top_indices])
            plt.yticks(range(top_n), [self.feature_names[i] for i in top_indices])
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'{self.model_name}: Top {top_n} Feature Importances', fontsize=14)
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
            plt.close()
            
            logger.info(f"\nTop 10 Features by Importance:")
            for i in range(min(10, len(self.feature_names))):
                idx = indices[i]
                logger.info(f"  {self.feature_names[idx]}: {importances[idx]:.4f}")
        else:
            logger.info("Model does not have feature_importances_ attribute")


# Main execution
if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    # Load best model
    model = joblib.load('models/random_forest.pkl')
    
    # Load feature names
    preprocessor = joblib.load('preprocessor.pkl')
    feature_names = preprocessor.feature_names
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, 'Random Forest', feature_names)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluator.evaluate_regression(X_test, y_test, y_pred)
    
    # Generate plots
    evaluator.plot_predictions(y_test, y_pred)
    evaluator.plot_residual_distribution(metrics['residuals'])
    evaluator.plot_error_by_range(y_test, y_pred)
    evaluator.feature_importance_sklearn()
    
    # SHAP analysis
    shap_values, importance_df = evaluator.shap_analysis(X_test, sample_size=1000)
    
    print("\n✓ Evaluation complete! Check the 'plots' folder for visualizations.")