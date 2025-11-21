# API Key Authentication Guide

## Quick Start

### **Step 1: Generate API Keys**

```bash
# Generate a new API key
python api_key_manager.py generate "Production Key" 1000

# This will output:
# ✅ Generated API Key:
#    abc123xyz456...
# 
# ⚠️  Save this key securely!
```

### **Step 2: Start the API Server**

```bash
python api_with_auth.py
```

### **Step 3: Use the API with Your Key**

#### **Using cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{
    "equipment_type": "CCTV Camera",
    "manufacturer": "Schneider",
    "model": "Schneider-120",
    "facility_type": "Academic",
    "facility_location": "Main Campus, Bangalore",
    "floor_count": 1,
    "total_area_sqm": 1200,
    "incident_category": "Network Outage",
    "year": 2023,
    "month": 10,
    "day_of_week": 5,
    "hour_of_day": 18,
    "is_weekend": 0,
    "season": "Autumn",
    "equipment_age_days": 1382.0,
    "severity": "Medium",
    "downtime_hours": 22.3,
    "cost_estimate": 25818.0,
    "days_since_maintenance": -735.0,
    "maintenance_cycle_days": 44
  }'
```

#### **Using Python:**
```python
import requests

api_key = "YOUR_API_KEY_HERE"
url = "http://localhost:5000/predict"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": api_key
}

data = {
    "equipment_type": "CCTV Camera",
    "manufacturer": "Schneider",
    # ... other fields
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result)
```

#### **Using JavaScript/Fetch:**
```javascript
const apiKey = "YOUR_API_KEY_HERE";
const url = "http://localhost:5000/predict";

const data = {
    equipment_type: "CCTV Camera",
    manufacturer: "Schneider",
    // ... other fields
};

fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => console.log(result))
.catch(error => console.error('Error:', error));
```

## API Key Methods

### **Method 1: Header (Recommended)**
```http
X-API-Key: YOUR_API_KEY_HERE
```

### **Method 2: Query Parameter**
```
http://localhost:5000/predict?api_key=YOUR_API_KEY_HERE
```

## Managing API Keys

### **Generate New Key:**
```bash
python api_key_manager.py generate "Client Name" 500
```

### **List All Keys:**
```bash
python api_key_manager.py list
```

### **Revoke a Key:**
```bash
python api_key_manager.py revoke abc123
```

## API Endpoints

### **1. Health Check** (No auth required)
```bash
GET http://localhost:5000/health
```

### **2. Make Prediction** (Requires API key)
```bash
POST http://localhost:5000/predict
Headers: X-API-Key: YOUR_KEY
Body: JSON with equipment data
```

### **3. Batch Predictions** (Requires API key)
```bash
POST http://localhost:5000/batch-predict
Headers: X-API-Key: YOUR_KEY
Body: {
  "items": [
    {equipment_data_1},
    {equipment_data_2},
    ...
  ]
}
```

### **4. Generate Key** (Admin only - add auth in production)
```bash
GET http://localhost:5000/generate-key
```

## Response Format

### **Success:**
```json
{
  "success": true,
  "prediction": {
    "days_to_failure": 87.64,
    "weeks_to_failure": 12.52,
    "months_to_failure": 2.92
  },
  "api_key_info": {
    "name": "Production Key",
    "rate_limit": 1000
  }
}
```

### **Error (Invalid Key):**
```json
{
  "success": false,
  "error": "Invalid API key"
}
```

### **Error (Missing Key):**
```json
{
  "success": false,
  "error": "API key required. Provide it in header: X-API-Key or query param: ?api_key=YOUR_KEY"
}
```

## Production Setup

### **1. Store Keys Securely**

Instead of hardcoding, use environment variables:

```python
import os

# In your .env file or environment
API_KEYS = {
    os.getenv('API_KEY_1'): {'name': 'Client 1', 'rate_limit': 1000},
    os.getenv('API_KEY_2'): {'name': 'Client 2', 'rate_limit': 500},
}
```

### **2. Use Database for Keys**

For production, store keys in a database:

```python
# Example with SQLite
import sqlite3

def get_api_key_info(api_key):
    conn = sqlite3.connect('api_keys.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT name, rate_limit, active FROM api_keys WHERE key = ?',
        (api_key,)
    )
    result = cursor.fetchone()
    conn.close()
    return result
```

### **3. Add Rate Limiting**

Use Flask-Limiter for advanced rate limiting:

```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day"]
)
```

### **4. Add HTTPS**

Always use HTTPS in production:

```python
# Use a production WSGI server
# gunicorn --bind 0.0.0.0:443 --certfile cert.pem --keyfile key.pem api_with_auth:app
```

## Integration Examples

### **Flask App Integration:**
```python
from flask import Flask, request
from model_predictor import TimeToFailurePredictor

app = Flask(__name__)
predictor = TimeToFailurePredictor()

# Your existing routes
@app.route('/your-endpoint')
def your_endpoint():
    api_key = request.headers.get('X-API-Key')
    if not validate_api_key(api_key):
        return {'error': 'Invalid API key'}, 401
    
    # Your logic here
    return {'result': 'success'}
```

### **Django Integration:**
```python
from django.http import JsonResponse
from model_predictor import TimeToFailurePredictor

predictor = TimeToFailurePredictor()

def predict_view(request):
    api_key = request.headers.get('X-API-Key')
    if not validate_api_key(api_key):
        return JsonResponse({'error': 'Invalid API key'}, status=401)
    
    data = request.POST
    result = predictor.predict(data)
    return JsonResponse({'days_to_failure': result})
```

## Security Best Practices

1. ✅ **Never commit API keys to Git** - Use environment variables
2. ✅ **Use HTTPS** - Always encrypt API communication
3. ✅ **Rotate keys regularly** - Generate new keys periodically
4. ✅ **Monitor usage** - Track API key usage for anomalies
5. ✅ **Set rate limits** - Prevent abuse
6. ✅ **Log requests** - Keep audit trail
7. ✅ **Use strong keys** - Generate with `secrets.token_urlsafe(32)`

## Testing Your API

```python
import requests

# Test with valid key
response = requests.post(
    'http://localhost:5000/predict',
    json={...your data...},
    headers={'X-API-Key': 'dev_key_12345'}
)
print(response.json())

# Test without key (should fail)
response = requests.post(
    'http://localhost:5000/predict',
    json={...your data...}
)
print(response.json())  # Should show error
```

