"""
NYC Taxi Data Loading Module
Connects to SQL database and loads data into pandas DataFrame
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import logging
from typing import Optional, Dict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxiDataLoader:
    """
    Handles connection to SQL database and data extraction
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize database connection
        
        Parameters:
        -----------
        connection_string : str
            SQLAlchemy connection string
            Examples:
            - PostgreSQL: 'postgresql://user:password@localhost:5432/taxidb'
            - MySQL: 'mysql+pymysql://user:password@localhost:3306/taxidb'
            - SQLite: 'sqlite:///taxi_data.db'
        """
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True  # Verify connections before using
        )
        logger.info("Database connection established")
    
    def load_data(
        self, 
        start_date: str, 
        end_date: str,
        limit: Optional[int] = None,
        sample_fraction: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Load taxi trip data from database with filters
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        limit : int, optional
            Maximum number of records to fetch
        sample_fraction : float, optional
            Fraction of data to sample (0.0 to 1.0)
        
        Returns:
        --------
        pd.DataFrame
            Loaded taxi trip data
        """
        
        # Build query with filters
        query = """
        SELECT 
            t.trip_id,
            t.vendor_id,
            t.pickup_datetime,
            t.dropoff_datetime,
            t.passenger_count,
            t.trip_distance,
            t.pickup_longitude,
            t.pickup_latitude,
            t.dropoff_longitude,
            t.dropoff_latitude,
            t.rate_code_id,
            t.store_and_fwd_flag,
            t.payment_type,
            t.fare_amount,
            t.extra,
            t.mta_tax,
            t.tip_amount,
            t.tolls_amount,
            t.improvement_surcharge,
            t.total_amount,
            t.congestion_surcharge,
            t.airport_fee
        FROM nyc_taxi_trips t
        WHERE t.pickup_datetime >= :start_date
          AND t.pickup_datetime < :end_date
          AND t.dropoff_datetime > t.pickup_datetime
          AND t.passenger_count BETWEEN 1 AND 6
          AND t.trip_distance BETWEEN 0.1 AND 100
          AND t.fare_amount BETWEEN 2.5 AND 500
          AND t.total_amount > 0
          AND t.pickup_latitude BETWEEN 40.5 AND 41.0
          AND t.pickup_longitude BETWEEN -74.5 AND -73.5
          AND t.dropoff_latitude BETWEEN 40.5 AND 41.0
          AND t.dropoff_longitude BETWEEN -74.5 AND -73.5
        """
        
        # Add sampling if specified
        if sample_fraction:
            query += f"\nAND RANDOM() < {sample_fraction}"
        
        # Add limit if specified
        if limit:
            query += f"\nLIMIT {limit}"
        
        logger.info(f"Loading data from {start_date} to {end_date}")
        
        try:
            # Execute query and load into DataFrame
            df = pd.read_sql_query(
                text(query),
                self.engine,
                params={'start_date': start_date, 'end_date': end_date},
                parse_dates=['pickup_datetime', 'dropoff_datetime']
            )
            
            logger.info(f"Loaded {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_table_stats(self) -> Dict:
        """
        Get summary statistics from the database
        
        Returns:
        --------
        Dict
            Dictionary containing table statistics
        """
        query = """
        SELECT 
            COUNT(*) as total_trips,
            MIN(pickup_datetime) as earliest_trip,
            MAX(pickup_datetime) as latest_trip,
            AVG(trip_distance) as avg_distance,
            AVG(fare_amount) as avg_fare,
            AVG(tip_amount) as avg_tip
        FROM nyc_taxi_trips
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
                stats = {
                    'total_trips': row[0],
                    'earliest_trip': row[1],
                    'latest_trip': row[2],
                    'avg_distance': float(row[3]) if row[3] else None,
                    'avg_fare': float(row[4]) if row[4] else None,
                    'avg_tip': float(row[5]) if row[5] else None
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()
        logger.info("Database connection closed")


# Example usage
if __name__ == "__main__":
    
    # Connection string examples for different databases
    # PostgreSQL
    # CONNECTION_STRING = 'postgresql://username:password@localhost:5432/taxi_db'
    
    # MySQL
    # CONNECTION_STRING = 'mysql+pymysql://username:password@localhost:3306/taxi_db'
    
    # SQLite (for testing)
    CONNECTION_STRING = 'sqlite:///nyc_taxi.db'
    
    # Initialize loader
    loader = TaxiDataLoader(CONNECTION_STRING)
    
    # Get database statistics
    print("Database Statistics:")
    stats = loader.get_table_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Load data for January 2024
    df = loader.load_data(
        start_date='2024-01-01',
        end_date='2024-02-01',
        limit=100000  # Limit for demonstration
    )
    
    print(f"\nLoaded DataFrame shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nBasic statistics:")
    print(df.describe())
    
    # Save to CSV for offline analysis
    df.to_csv('taxi_data_raw.csv', index=False)
    print("\nData saved to 'taxi_data_raw.csv'")
    
    # Close connection
    loader.close()