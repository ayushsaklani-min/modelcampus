# How to Start the API Server

## Step 1: Navigate to the Model Directory

```powershell
cd C:\Users\ankur\Downloads\12\12
```

## Step 2: Start the API Server

```powershell
python api_with_auth.py
```

You should see:
```
============================================================
API Server Starting...
============================================================

Available API Keys:
  PfeQLfnlHgUKrdpRoB3E7kpGdOaKo-kkHXGeFg5Iyzw... - API Key

To use the API, include header: X-API-Key: YOUR_KEY
Or use query parameter: ?api_key=YOUR_KEY

 * Running on http://0.0.0.0:5000
```

## Step 3: Keep the Server Running

**Keep this terminal window open!** The server needs to stay running.

## Step 4: Test in Another Terminal

Open a **new terminal window** and run your test:

```powershell
cd C:\Users\ankur\OneDrive\Desktop\help_help2
node test-api-correct.js
```

## Troubleshooting

### Error: "ECONNREFUSED"
- **Solution**: Make sure the API server is running (Step 2)

### Error: "Invalid API key"
- **Solution**: Check that your API key matches the one in `api_keys.json`
- List keys: `python api_key_manager.py list`

### Error: "Module not found"
- **Solution**: Install dependencies: `pip install flask pandas numpy scikit-learn joblib`

### Port Already in Use
- **Solution**: Change port in `api_with_auth.py` (last line: `app.run(..., port=5001)`) or kill the process using port 5000

## Correct Data Format

The API expects equipment data, NOT location/category data:

```javascript
// ✅ CORRECT
{
    equipment_type: 'CCTV Camera',
    manufacturer: 'Schneider',
    facility_type: 'Academic',
    equipment_age_days: 1382.0,
    month: 10,
    year: 2023,
    // ... see test-api-correct.js for full format
}

// ❌ WRONG (your current format)
{
    location: 'Building A',
    category: 'water',
    historical_data: [...]
}
```

## Quick Test

Use the provided `test-api-correct.js` file which has the correct format!

