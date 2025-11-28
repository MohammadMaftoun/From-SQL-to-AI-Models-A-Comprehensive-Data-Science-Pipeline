-- =====================================================
-- NYC TAXI TRIP DATA - SQL SCHEMA & EXTRACTION QUERIES
-- =====================================================

-- 1. CREATE TABLE SCHEMA
-- -----------------------------------------------------
CREATE TABLE nyc_taxi_trips (
    trip_id BIGINT PRIMARY KEY,
    vendor_id INT NOT NULL,
    pickup_datetime TIMESTAMP NOT NULL,
    dropoff_datetime TIMESTAMP NOT NULL,
    passenger_count INT,
    trip_distance DECIMAL(10, 2),
    pickup_longitude DECIMAL(10, 6),
    pickup_latitude DECIMAL(10, 6),
    dropoff_longitude DECIMAL(10, 6),
    dropoff_latitude DECIMAL(10, 6),
    rate_code_id INT,
    store_and_fwd_flag CHAR(1),
    payment_type INT,
    fare_amount DECIMAL(10, 2),
    extra DECIMAL(10, 2),
    mta_tax DECIMAL(10, 2),
    tip_amount DECIMAL(10, 2),
    tolls_amount DECIMAL(10, 2),
    improvement_surcharge DECIMAL(10, 2),
    total_amount DECIMAL(10, 2),
    congestion_surcharge DECIMAL(10, 2),
    airport_fee DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pickup_datetime (pickup_datetime),
    INDEX idx_dropoff_datetime (dropoff_datetime),
    INDEX idx_payment_type (payment_type)
);

-- Reference table for rate codes
CREATE TABLE rate_codes (
    rate_code_id INT PRIMARY KEY,
    rate_code_name VARCHAR(50)
);

INSERT INTO rate_codes VALUES
(1, 'Standard rate'),
(2, 'JFK'),
(3, 'Newark'),
(4, 'Nassau or Westchester'),
(5, 'Negotiated fare'),
(6, 'Group ride');

-- Reference table for payment types
CREATE TABLE payment_types (
    payment_type INT PRIMARY KEY,
    payment_name VARCHAR(50)
);

INSERT INTO payment_types VALUES
(1, 'Credit card'),
(2, 'Cash'),
(3, 'No charge'),
(4, 'Dispute'),
(5, 'Unknown'),
(6, 'Voided trip');


-- 2. BASIC DATA EXTRACTION QUERY
-- -----------------------------------------------------
-- Extract all relevant columns for modeling
SELECT 
    trip_id,
    vendor_id,
    pickup_datetime,
    dropoff_datetime,
    passenger_count,
    trip_distance,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    rate_code_id,
    store_and_fwd_flag,
    payment_type,
    fare_amount,
    extra,
    mta_tax,
    tip_amount,
    tolls_amount,
    improvement_surcharge,
    total_amount,
    congestion_surcharge,
    airport_fee
FROM nyc_taxi_trips
WHERE pickup_datetime >= '2024-01-01'
  AND pickup_datetime < '2024-02-01'
  AND trip_distance > 0
  AND fare_amount > 0
LIMIT 100000;


-- 3. FILTERED QUERY WITH DATA QUALITY CHECKS
-- -----------------------------------------------------
-- Extract clean data with basic quality filters
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
    rc.rate_code_name,
    t.store_and_fwd_flag,
    t.payment_type,
    pt.payment_name,
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
LEFT JOIN rate_codes rc ON t.rate_code_id = rc.rate_code_id
LEFT JOIN payment_types pt ON t.payment_type = pt.payment_type
WHERE t.pickup_datetime >= '2024-01-01'
  AND t.pickup_datetime < '2024-02-01'
  AND t.dropoff_datetime > t.pickup_datetime
  AND t.passenger_count BETWEEN 1 AND 6
  AND t.trip_distance BETWEEN 0.1 AND 100
  AND t.fare_amount BETWEEN 2.5 AND 500
  AND t.total_amount BETWEEN 2.5 AND 500
  AND t.pickup_latitude BETWEEN 40.5 AND 41.0
  AND t.pickup_longitude BETWEEN -74.5 AND -73.5
  AND t.dropoff_latitude BETWEEN 40.5 AND 41.0
  AND t.dropoff_longitude BETWEEN -74.5 AND -73.5
LIMIT 100000;


-- 4. AGGREGATED STATISTICS QUERY
-- -----------------------------------------------------
-- Get summary statistics for exploratory analysis
SELECT 
    COUNT(*) as total_trips,
    AVG(trip_distance) as avg_distance,
    AVG(fare_amount) as avg_fare,
    AVG(tip_amount) as avg_tip,
    AVG(passenger_count) as avg_passengers,
    MIN(pickup_datetime) as earliest_trip,
    MAX(pickup_datetime) as latest_trip,
    COUNT(DISTINCT vendor_id) as unique_vendors,
    SUM(CASE WHEN payment_type = 1 THEN 1 ELSE 0 END) as credit_card_trips,
    SUM(CASE WHEN payment_type = 2 THEN 1 ELSE 0 END) as cash_trips
FROM nyc_taxi_trips
WHERE pickup_datetime >= '2024-01-01'
  AND pickup_datetime < '2024-02-01';


-- 5. TIME-BASED ANALYSIS QUERY
-- -----------------------------------------------------
-- Extract trips with time features for peak hour analysis
SELECT 
    trip_id,
    pickup_datetime,
    EXTRACT(HOUR FROM pickup_datetime) as pickup_hour,
    EXTRACT(DOW FROM pickup_datetime) as pickup_dow,
    CASE 
        WHEN EXTRACT(DOW FROM pickup_datetime) IN (0, 6) THEN 1 
        ELSE 0 
    END as is_weekend,
    trip_distance,
    fare_amount,
    tip_amount,
    passenger_count,
    payment_type
FROM nyc_taxi_trips
WHERE pickup_datetime >= '2024-01-01'
  AND pickup_datetime < '2024-02-01'
  AND trip_distance > 0;


-- 6. SAMPLING QUERY FOR LARGE DATASETS
-- -----------------------------------------------------
-- Use random sampling for very large datasets
SELECT 
    trip_id,
    vendor_id,
    pickup_datetime,
    dropoff_datetime,
    passenger_count,
    trip_distance,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    rate_code_id,
    payment_type,
    fare_amount,
    tip_amount,
    total_amount
FROM nyc_taxi_trips
WHERE pickup_datetime >= '2024-01-01'
  AND pickup_datetime < '2024-07-01'
  AND RANDOM() < 0.1  -- 10% sample (PostgreSQL syntax)
  -- For MySQL: AND RAND() < 0.1
  -- For SQL Server: ORDER BY NEWID()
LIMIT 500000;