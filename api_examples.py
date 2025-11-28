"""
API Testing Examples and Documentation
Demonstrates how to interact with the NYC Taxi Tip Prediction API
"""

import requests
import json
from typing import Dict, List

# ==================== Configuration ====================

API_BASE_URL = "http://localhost:8000"


# ==================== Helper Functions ====================

def print_response(response: requests.Response, title: str):
    """
    Pretty print API response
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse:")
    print(json.dumps(response.json(), indent=2))


# ==================== Test Functions ====================

def test_health_check():
    """
    Test health check endpoint
    """
    response = requests.get(f"{API_BASE_URL}/health")
    print_response(response, "Health Check")


def test_single_prediction():
    """
    Test single trip prediction
    """
    trip_data = {
        "pickup_datetime": "2024-01-15 14:30:00",
        "dropoff_datetime": "2024-01-15 14:45:00",
        "passenger_count": 2,
        "trip_distance": 5.2,
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.95,
        "dropoff_latitude": 40.77,
        "fare_amount": 15.50,
        "payment_type": 1,
        "rate_code_id": 1,
        "vendor_id": 1,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 2.5,
        "airport_fee": 0.0,
        "store_and_fwd_flag": "N"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=trip_data,
        headers={"Content-Type": "application/json"}
    )
    
    print_response(response, "Single Trip Prediction")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n💡 Insights:")
        print(f"   For a ${result['fare_amount']:.2f} fare")
        print(f"   Expected tip: ${result['predicted_tip']:.2f}")
        print(f"   Tip percentage: {result['tip_percentage']:.1f}%")


def test_batch_prediction():
    """
    Test batch prediction
    """
    trips = {
        "trips": [
            {
                "pickup_datetime": "2024-01-15 08:00:00",
                "dropoff_datetime": "2024-01-15 08:20:00",
                "passenger_count": 1,
                "trip_distance": 3.5,
                "pickup_longitude": -73.98,
                "pickup_latitude": 40.75,
                "dropoff_longitude": -73.95,
                "dropoff_latitude": 40.77,
                "fare_amount": 12.00,
                "payment_type": 1,
                "rate_code_id": 1,
                "vendor_id": 1,
                "extra": 0.0,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 0.3,
                "congestion_surcharge": 2.5,
                "airport_fee": 0.0,
                "store_and_fwd_flag": "N"
            },
            {
                "pickup_datetime": "2024-01-15 18:30:00",
                "dropoff_datetime": "2024-01-15 19:00:00",
                "passenger_count": 3,
                "trip_distance": 8.2,
                "pickup_longitude": -73.99,
                "pickup_latitude": 40.74,
                "dropoff_longitude": -73.92,
                "dropoff_latitude": 40.78,
                "fare_amount": 28.50,
                "payment_type": 1,
                "rate_code_id": 1,
                "vendor_id": 2,
                "extra": 1.0,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 0.3,
                "congestion_surcharge": 2.5,
                "airport_fee": 0.0,
                "store_and_fwd_flag": "N"
            }
        ]
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict/batch",
        json=trips,
        headers={"Content-Type": "application/json"}
    )
    
    print_response(response, "Batch Prediction")


def test_model_info():
    """
    Test model info endpoint
    """
    response = requests.get(f"{API_BASE_URL}/model/info")
    print_response(response, "Model Information")


def test_edge_cases():
    """
    Test edge cases and error handling
    """
    print(f"\n{'='*60}")
    print("Testing Edge Cases")
    print(f"{'='*60}")
    
    # Test 1: Invalid datetime format
    print("\n1. Invalid datetime format:")
    invalid_trip = {
        "pickup_datetime": "invalid-date",
        "dropoff_datetime": "2024-01-15 14:45:00",
        "passenger_count": 2,
        "trip_distance": 5.2,
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.95,
        "dropoff_latitude": 40.77,
        "fare_amount": 15.50,
        "payment_type": 1
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=invalid_trip)
    print(f"Status Code: {response.status_code}")
    print(f"Error: {response.json()}")
    
    # Test 2: Out of range values
    print("\n2. Out of range passenger count:")
    invalid_trip2 = {
        "pickup_datetime": "2024-01-15 14:30:00",
        "dropoff_datetime": "2024-01-15 14:45:00",
        "passenger_count": 10,  # Invalid: > 6
        "trip_distance": 5.2,
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.95,
        "dropoff_latitude": 40.77,
        "fare_amount": 15.50,
        "payment_type": 1
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=invalid_trip2)
    print(f"Status Code: {response.status_code}")
    print(f"Error: {response.json()}")


# ==================== Example JSON Requests ====================

EXAMPLE_CURL_SINGLE = """
# Example cURL command for single prediction:

curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "pickup_datetime": "2024-01-15 14:30:00",
    "dropoff_datetime": "2024-01-15 14:45:00",
    "passenger_count": 2,
    "trip_distance": 5.2,
    "pickup_longitude": -73.98,
    "pickup_latitude": 40.75,
    "dropoff_longitude": -73.95,
    "dropoff_latitude": 40.77,
    "fare_amount": 15.50,
    "payment_type": 1,
    "rate_code_id": 1,
    "vendor_id": 1,
    "extra": 0.5,
    "mta_tax": 0.5,
    "tolls_amount": 0.0,
    "improvement_surcharge": 0.3,
    "congestion_surcharge": 2.5,
    "airport_fee": 0.0,
    "store_and_fwd_flag": "N"
  }'
"""

EXAMPLE_CURL_BATCH = """
# Example cURL command for batch prediction:

curl -X POST "http://localhost:8000/predict/batch" \\
  -H "Content-Type: application/json" \\
  -d '{
    "trips": [
      {
        "pickup_datetime": "2024-01-15 08:00:00",
        "dropoff_datetime": "2024-01-15 08:20:00",
        "passenger_count": 1,
        "trip_distance": 3.5,
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.95,
        "dropoff_latitude": 40.77,
        "fare_amount": 12.00,
        "payment_type": 1
      },
      {
        "pickup_datetime": "2024-01-15 18:30:00",
        "dropoff_datetime": "2024-01-15 19:00:00",
        "passenger_count": 3,
        "trip_distance": 8.2,
        "pickup_longitude": -73.99,
        "pickup_latitude": 40.74,
        "dropoff_longitude": -73.92,
        "dropoff_latitude": 40.78,
        "fare_amount": 28.50,
        "payment_type": 1
      }
    ]
  }'
"""

EXAMPLE_RESPONSE = """
# Example Response:

{
  "predicted_tip": 3.25,
  "confidence_interval": {
    "lower": 1.75,
    "upper": 4.75
  },
  "fare_amount": 15.50,
  "tip_percentage": 20.97,
  "model_version": "1.0.0"
}
"""


# ==================== JavaScript Example ====================

JAVASCRIPT_EXAMPLE = """
// JavaScript/Node.js Example:

const axios = require('axios');

async function predictTip() {
  const tripData = {
    pickup_datetime: "2024-01-15 14:30:00",
    dropoff_datetime: "2024-01-15 14:45:00",
    passenger_count: 2,
    trip_distance: 5.2,
    pickup_longitude: -73.98,
    pickup_latitude: 40.75,
    dropoff_longitude: -73.95,
    dropoff_latitude: 40.77,
    fare_amount: 15.50,
    payment_type: 1,
    rate_code_id: 1,
    vendor_id: 1,
    extra: 0.5,
    mta_tax: 0.5,
    tolls_amount: 0.0,
    improvement_surcharge: 0.3,
    congestion_surcharge: 2.5,
    airport_fee: 0.0,
    store_and_fwd_flag: "N"
  };

  try {
    const response = await axios.post(
      'http://localhost:8000/predict',
      tripData
    );
    
    console.log('Predicted tip:', response.data.predicted_tip);
    console.log('Tip percentage:', response.data.tip_percentage);
  } catch (error) {
    console.error('Error:', error.response.data);
  }
}

predictTip();
"""


# ==================== Main Execution ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NYC Taxi Tip Prediction API - Test Suite")
    print("="*60)
    
    try:
        # Run all tests
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        test_edge_cases()
        
        # Print examples
        print("\n" + "="*60)
        print("cURL Examples")
        print("="*60)
        print(EXAMPLE_CURL_SINGLE)
        print(EXAMPLE_CURL_BATCH)
        print("\n" + "="*60)
        print("Example Response")
        print("="*60)
        print(EXAMPLE_RESPONSE)
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("   Make sure the API server is running:")
        print("   python api.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")