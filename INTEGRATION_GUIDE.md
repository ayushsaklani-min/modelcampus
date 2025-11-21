# Integration Guide: Using ML Model in Your Web App

## Step-by-Step Instructions

### **Step 1: Copy Model Files to Your Web App**

Copy these files from the model directory to your web app project:
- `best_model.pkl`
- `label_encoders.pkl`
- `scaler.pkl`
- `feature_info.pkl`
- `model_predictor.py` (the module we created)

**Example structure:**
```
your-web-app/
├── models/
│   ├── best_model.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── feature_info.pkl
├── model_predictor.py
└── your-app-files...
```

### **Step 2: Install Dependencies**

Add to your `requirements.txt` or install:
```bash
pip install pandas numpy scikit-learn joblib
```

### **Step 3: Import and Initialize in Your App**

#### **For Flask:**
```python
from flask import Flask, request, jsonify
from model_predictor import TimeToFailurePredictor

app = Flask(__name__)

# Initialize predictor (loads model once at startup)
predictor = TimeToFailurePredictor(model_dir='./models')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Make prediction
    days_to_failure = predictor.predict(data)
    
    return jsonify({
        'days_to_failure': round(days_to_failure, 2),
        'message': f'Predicted time to failure: {days_to_failure:.2f} days'
    })
```

#### **For Django:**
```python
# views.py
from django.http import JsonResponse
from model_predictor import TimeToFailurePredictor
import os

# Initialize once (can be module-level or in app config)
predictor = TimeToFailurePredictor(
    model_dir=os.path.join(os.path.dirname(__file__), 'models')
)

def predict_failure(request):
    if request.method == 'POST':
        data = request.POST.dict() or request.body
        
        days_to_failure = predictor.predict(data)
        
        return JsonResponse({
            'days_to_failure': round(days_to_failure, 2)
        })
```

#### **For FastAPI:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
from model_predictor import TimeToFailurePredictor

app = FastAPI()

# Initialize predictor
predictor = TimeToFailurePredictor(model_dir='./models')

class PredictionInput(BaseModel):
    equipment_type: str
    manufacturer: str
    model: str
    facility_type: str
    # ... add all required fields

@app.post("/predict")
def predict(input_data: PredictionInput):
    days_to_failure = predictor.predict(input_data.dict())
    return {"days_to_failure": round(days_to_failure, 2)}
```

#### **For Node.js/Express (using Python subprocess):**
```javascript
const express = require('express');
const { spawn } = require('child_process');
const app = express();

app.post('/predict', (req, res) => {
    const python = spawn('python', ['predict_api.py', JSON.stringify(req.body)]);
    
    python.stdout.on('data', (data) => {
        const result = JSON.parse(data.toString());
        res.json(result);
    });
    
    python.stderr.on('data', (data) => {
        res.status(500).json({ error: data.toString() });
    });
});
```

Create `predict_api.py`:
```python
import sys
import json
from model_predictor import TimeToFailurePredictor

predictor = TimeToFailurePredictor(model_dir='./models')
data = json.loads(sys.argv[1])
result = predictor.predict(data)
print(json.dumps({'days_to_failure': round(result, 2)}))
```

### **Step 4: Prepare Input Data**

Your input data must include these fields:

**Required Fields:**
```python
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
```

### **Step 5: Make Predictions**

#### **Single Prediction:**
```python
days = predictor.predict(input_data)
print(f"Days to failure: {days}")
```

#### **Batch Predictions:**
```python
multiple_inputs = [input_data1, input_data2, input_data3]
predictions = predictor.predict_batch(multiple_inputs)
```

### **Step 6: Handle Errors**

```python
try:
    days_to_failure = predictor.predict(input_data)
    return {'success': True, 'days_to_failure': days_to_failure}
except Exception as e:
    return {'success': False, 'error': str(e)}
```

## Complete Example: Flask Integration

```python
from flask import Flask, request, jsonify, render_template
from model_predictor import TimeToFailurePredictor

app = Flask(__name__)

# Initialize once at startup
predictor = TimeToFailurePredictor(model_dir='./models')

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['equipment_type', 'manufacturer', 'facility_type', 
                          'equipment_age_days', 'month', 'year']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
        
        # Make prediction
        days_to_failure = predictor.predict(data)
        
        return jsonify({
            'success': True,
            'days_to_failure': round(days_to_failure, 2),
            'weeks_to_failure': round(days_to_failure / 7, 2),
            'months_to_failure': round(days_to_failure / 30, 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Frontend Example (JavaScript)

```javascript
async function predictFailure(formData) {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log(`Days to failure: ${result.days_to_failure}`);
            return result;
        } else {
            console.error('Prediction error:', result.error);
            return null;
        }
    } catch (error) {
        console.error('Request failed:', error);
        return null;
    }
}

// Usage
const inputData = {
    equipment_type: 'CCTV Camera',
    manufacturer: 'Schneider',
    // ... other fields
};

predictFailure(inputData).then(result => {
    if (result) {
        document.getElementById('result').textContent = 
            `Predicted: ${result.days_to_failure} days`;
    }
});
```

## Important Notes

1. **Model Loading**: Initialize the predictor once at app startup, not on every request
2. **Feature Order**: The model expects features in a specific order - the `model_predictor.py` handles this automatically
3. **Missing Values**: Handle missing values in your input before prediction
4. **Data Types**: Ensure numerical fields are numbers, not strings
5. **Categorical Values**: Use the same categorical values seen during training (check your CSV for valid options)

## Testing

Test with sample data:
```python
from model_predictor import TimeToFailurePredictor

predictor = TimeToFailurePredictor(model_dir='./models')

test_data = {
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

result = predictor.predict(test_data)
print(f"Predicted days to failure: {result}")
```

