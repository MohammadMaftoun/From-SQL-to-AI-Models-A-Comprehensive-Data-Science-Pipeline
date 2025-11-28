"""
Complete Data Preprocessing and Feature Engineering Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
import logging
from typing import Tuple, List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxiDataPreprocessor:
    """
    Comprehensive preprocessing and feature engineering for taxi data
    """
    
    def __init__(self):
        self.scaler = RobustScaler()  # Robust to outliers
        self.label_encoders = {}
        self.feature_names = None
        self.target_column = 'tip_amount'
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        """
        logger.info("Handling missing values...")
        
        # Log missing value counts
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # Strategy for each column type
        # Numerical columns: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {col} with median: {median_val}")
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled {col} with mode: {mode_val}")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using IQR method and domain knowledge
        """
        logger.info("Removing outliers...")
        initial_len = len(df)
        
        # Domain-specific outlier removal
        df = df[
            (df['trip_distance'] > 0.1) & (df['trip_distance'] < 100) &
            (df['fare_amount'] > 0) & (df['fare_amount'] < 500) &
            (df['tip_amount'] >= 0) & (df['tip_amount'] < 200) &
            (df['total_amount'] > 0) & (df['total_amount'] < 500) &
            (df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)
        ]
        
        # IQR-based outlier removal for key features
        def remove_outliers_iqr(data, column, factor=1.5):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        
        # Apply IQR to selected columns
        for col in ['trip_distance', 'fare_amount', 'tip_amount']:
            df = remove_outliers_iqr(df, col, factor=2.0)
        
        removed = initial_len - len(df)
        logger.info(f"Removed {removed} outliers ({removed/initial_len*100:.2f}%)")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data
        """
        logger.info("Engineering features...")
        
        # 1. Trip duration (in minutes)
        df['trip_duration_minutes'] = (
            df['dropoff_datetime'] - df['pickup_datetime']
        ).dt.total_seconds() / 60
        
        # Remove invalid durations
        df = df[
            (df['trip_duration_minutes'] > 0) & 
            (df['trip_duration_minutes'] < 180)  # Less than 3 hours
        ]
        
        # 2. Speed (miles per hour)
        df['speed_mph'] = df['trip_distance'] / (df['trip_duration_minutes'] / 60)
        df['speed_mph'] = df['speed_mph'].replace([np.inf, -np.inf], 0)
        df['speed_mph'] = df['speed_mph'].clip(0, 100)  # Cap at reasonable speed
        
        # 3. Tip percentage
        df['tip_percentage'] = (df['tip_amount'] / df['fare_amount'] * 100).clip(0, 100)
        
        # 4. Time-based features
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['pickup_day'] = df['pickup_datetime'].dt.day
        df['pickup_month'] = df['pickup_datetime'].dt.month
        
        # 5. Categorical time features
        df['is_weekend'] = (df['pickup_day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df['is_night'] = df['pickup_hour'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
        
        # 6. Time of day categories
        def categorize_time(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'
        
        df['time_of_day'] = df['pickup_hour'].apply(categorize_time)
        
        # 7. Distance features
        df['haversine_distance'] = self._calculate_haversine(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        
        # 8. Location features (simplified - Manhattan grid)
        df['pickup_location_cluster'] = self._location_cluster(
            df['pickup_latitude'], df['pickup_longitude']
        )
        df['dropoff_location_cluster'] = self._location_cluster(
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        
        # 9. Fare-related features
        df['fare_per_mile'] = (df['fare_amount'] / df['trip_distance']).replace([np.inf, -np.inf], 0)
        df['fare_per_minute'] = (df['fare_amount'] / df['trip_duration_minutes']).replace([np.inf, -np.inf], 0)
        
        # 10. Payment type indicator (credit card more likely to tip)
        df['is_credit_card'] = (df['payment_type'] == 1).astype(int)
        
        # 11. Surge pricing indicator (high fare per mile)
        df['is_surge'] = (df['fare_per_mile'] > df['fare_per_mile'].quantile(0.75)).astype(int)
        
        # 12. Airport trips
        df['is_airport_trip'] = df['rate_code_id'].isin([2, 3]).astype(int)
        
        logger.info(f"Created {len(df.columns)} total features")
        
        return df
    
    def _calculate_haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate haversine distance between two points
        """
        R = 3959.87433  # Radius of Earth in miles
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _location_cluster(self, lat, lon):
        """
        Simple grid-based location clustering
        """
        # Create 10x10 grid of NYC
        lat_bins = pd.cut(lat, bins=10, labels=False)
        lon_bins = pd.cut(lon, bins=10, labels=False)
        return lat_bins * 10 + lon_bins
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        """
        logger.info("Encoding categorical variables...")
        
        # Columns to encode
        categorical_cols = ['time_of_day', 'store_and_fwd_flag']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str, method='mutual_info', k=20) -> List[str]:
        """
        Select top k features using mutual information or random forest
        """
        logger.info(f"Selecting top {k} features using {method}...")
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = [
            target_col, 'trip_id', 'pickup_datetime', 'dropoff_datetime',
            'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'time_of_day', 'store_and_fwd_flag'  # Already encoded
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        if method == 'mutual_info':
            # Mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
            selected_features = mi_scores.head(k).index.tolist()
            
            logger.info("\nTop 10 features by Mutual Information:")
            for feat, score in mi_scores.head(10).items():
                logger.info(f"  {feat}: {score:.4f}")
        
        elif method == 'random_forest':
            # Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
            selected_features = importances.head(k).index.tolist()
            
            logger.info("\nTop 10 features by Random Forest Importance:")
            for feat, score in importances.head(10).items():
                logger.info(f"  {feat}: {score:.4f}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return selected_features
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale features using RobustScaler
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'tip_amount') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Complete preprocessing pipeline
        """
        self.target_column = target_col
        
        # 1. Handle missing values
        df = self.handle_missing_values(df.copy())
        
        # 2. Remove outliers
        df = self.remove_outliers(df)
        
        # 3. Engineer features
        df = self.engineer_features(df)
        
        # 4. Encode categorical
        df = self.encode_categorical(df)
        
        # 5. Select features
        selected_features = self.select_features(df, target_col, method='random_forest', k=20)
        self.feature_names = selected_features
        
        # 6. Prepare X and y
        X = df[selected_features].fillna(0)
        y = df[target_col].values
        
        # 7. Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        logger.info(f"\nFinal dataset shape: X={X_scaled.shape}, y={y.shape}")
        
        return X_scaled, y, selected_features
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor
        """
        df = self.handle_missing_values(df.copy())
        df = self.engineer_features(df)
        df = self.encode_categorical(df)
        
        X = df[self.feature_names].fillna(0)
        X_scaled = self.scale_features(X, fit=False)
        
        return X_scaled


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('taxi_data_raw.csv', parse_dates=['pickup_datetime', 'dropoff_datetime'])
    
    print(f"Raw data shape: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = TaxiDataPreprocessor()
    
    # Fit and transform
    X, y, features = preprocessor.fit_transform(df, target_col='tip_amount')
    
    print(f"\nProcessed data shape: X={X.shape}, y={y.shape}")
    print(f"\nSelected features: {features}")
    
    # Save processed data
    np.save('X_processed.npy', X)
    np.save('y_processed.npy', y)
    
    # Save preprocessor (using joblib)
    import joblib
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    print("\nPreprocessing complete!")