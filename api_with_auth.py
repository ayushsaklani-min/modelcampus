"""
Flask API with API Key Authentication
Use this to secure your prediction endpoint
"""

from flask import Flask, request, jsonify
from functools import wraps
from model_predictor import TimeToFailurePredictor
import os
import secrets

app = Flask(__name__)

# Initialize predictor
predictor = TimeToFailurePredictor(model_dir='.')

# API Key Management
# Load keys from api_keys.json file (created by api_key_manager.py)
API_KEYS = {}

# Load from api_keys.json if it exists
if os.path.exists('api_keys.json'):
    import json
    with open('api_keys.json', 'r') as f:
        api_keys_data = json.load(f)
        # Convert the format: {key: {metadata}} -> {key: {name, rate_limit}}
        for key, metadata in api_keys_data.items():
            API_KEYS[key] = {
                'name': metadata.get('name', 'Unknown'),
                'rate_limit': metadata.get('rate_limit', 100)
            }

# Fallback to hardcoded keys if no file exists
if not API_KEYS:
    API_KEYS = {
        'dev_key_12345': {'name': 'Development Key', 'rate_limit': 100},
        'prod_key_67890': {'name': 'Production Key', 'rate_limit': 1000},
    }

# Or load from environment variable
if os.getenv('API_KEYS'):
    import json
    env_keys = json.loads(os.getenv('API_KEYS'))
    for key, value in env_keys.items():
        if isinstance(value, dict):
            API_KEYS[key] = value
        else:
            API_KEYS[key] = {'name': 'Env Key', 'rate_limit': 100}


def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API key required. Provide it in header: X-API-Key or query param: ?api_key=YOUR_KEY'
            }), 401
        
        if api_key not in API_KEYS:
            return jsonify({
                'success': False,
                'error': 'Invalid API key'
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    return jsonify({
        'message': 'Time-to-Failure Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Make predictions (requires API key)',
            '/health': 'GET - Check API health',
            '/generate-key': 'GET - Generate new API key (admin only)'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None
    })


@app.route('/generate-key')
def generate_key():
    """Generate a new API key (in production, add admin authentication)"""
    new_key = secrets.token_urlsafe(32)
    API_KEYS[new_key] = {
        'name': f'Generated Key',
        'rate_limit': 100
    }
    return jsonify({
        'api_key': new_key,
        'message': 'Save this key securely. It will not be shown again.'
    })


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """Prediction endpoint with API key authentication"""
    try:
        # Get API key info
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        key_info = API_KEYS.get(api_key, {})
        
        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided. Send JSON in request body.'
            }), 400
        
        # Validate required fields
        required_fields = [
            'equipment_type', 'manufacturer', 'facility_type',
            'equipment_age_days', 'month', 'year'
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing}'
            }), 400
        
        # Make prediction
        days_to_failure = predictor.predict(data)
        
        return jsonify({
            'success': True,
            'prediction': {
                'days_to_failure': round(days_to_failure, 2),
                'weeks_to_failure': round(days_to_failure / 7, 2),
                'months_to_failure': round(days_to_failure / 30, 2)
            },
            'api_key_info': {
                'name': key_info.get('name', 'Unknown'),
                'rate_limit': key_info.get('rate_limit', 'unlimited')
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch-predict', methods=['POST'])
@require_api_key
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'items' not in data:
            return jsonify({
                'success': False,
                'error': 'Expected JSON with "items" array'
            }), 400
        
        items = data['items']
        if not isinstance(items, list):
            return jsonify({
                'success': False,
                'error': '"items" must be an array'
            }), 400
        
        # Make batch predictions
        predictions = predictor.predict_batch(items)
        
        results = [
            {
                'days_to_failure': round(p, 2),
                'weeks_to_failure': round(p / 7, 2),
                'months_to_failure': round(p / 30, 2)
            }
            for p in predictions
        ]
        
        return jsonify({
            'success': True,
            'count': len(results),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("API Server Starting...")
    print("=" * 60)
    print("\nAvailable API Keys:")
    for key, info in API_KEYS.items():
        print(f"  {key[:20]}... - {info.get('name', 'Unknown')}")
    print("\nTo use the API, include header: X-API-Key: YOUR_KEY")
    print("Or use query parameter: ?api_key=YOUR_KEY")
    print("\nExample:")
    print("  curl -X POST http://localhost:5000/predict \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -H 'X-API-Key: dev_key_12345' \\")
    print("    -d '{\"equipment_type\": \"CCTV Camera\", ...}'")
    print("\n" + "=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

