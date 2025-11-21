"""
Example: How to use the model in your web app
"""

from model_predictor import TimeToFailurePredictor

# Step 1: Initialize the predictor (do this once at app startup)
predictor = TimeToFailurePredictor(model_dir='.')

# Step 2: Prepare your input data (from form, API request, etc.)
input_data = {
    'equipment_type': 'CCTV Camera',
    'manufacturer': 'Schneider',
    'model': 'Schneider-120',
    'facility_type': 'Academic',
    'facility_location': 'Main Campus, Bangalore',
    'floor_count': 1,
    'total_area_sqm': 1200,
    'incident_category': 'Network Outage',
    'year': 2023,
    'month': 10,
    'day_of_week': 5,
    'hour_of_day': 18,
    'is_weekend': 0,
    'season': 'Autumn',
    'equipment_age_days': 1382.0,
    'severity': 'Medium',
    'downtime_hours': 22.3,
    'cost_estimate': 25818.0,
    'days_since_maintenance': -735.0,
    'maintenance_cycle_days': 44
}

# Step 3: Make prediction
try:
    days_to_failure = predictor.predict(input_data)
    print(f"✅ Predicted days to failure: {days_to_failure:.2f} days")
    print(f"   That's approximately {days_to_failure/7:.1f} weeks or {days_to_failure/30:.1f} months")
except Exception as e:
    print(f"❌ Error: {e}")

# Example: Batch predictions
multiple_equipment = [
    {
        'equipment_type': 'Fire Alarm',
        'manufacturer': 'Emerson',
        'model': 'Emerson-356',
        'facility_type': 'Hostel',
        'facility_location': 'East Campus, Bangalore',
        'floor_count': 1,
        'total_area_sqm': 600,
        'incident_category': 'IoT Sensor Fault',
        'year': 2023,
        'month': 8,
        'day_of_week': 5,
        'hour_of_day': 7,
        'is_weekend': 0,
        'season': 'Monsoon',
        'equipment_age_days': 2779.0,
        'severity': 'High',
        'downtime_hours': 13.1,
        'cost_estimate': 47671.0,
        'days_since_maintenance': -684.0,
        'maintenance_cycle_days': 164
    },
    # Add more equipment data...
]

try:
    predictions = predictor.predict_batch(multiple_equipment)
    print(f"\n✅ Batch predictions: {predictions}")
except Exception as e:
    print(f"❌ Batch prediction error: {e}")

