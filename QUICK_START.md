# Quick Start Guide

## ✅ Step 1: Install Dependencies (Already Done!)
All dependencies are installed. You're ready to go!

## 🚀 Step 2: Start the API Server

**Option A: Double-click the batch file**
- Double-click `start_api.bat` in the `12` folder

**Option B: Manual start**
Open PowerShell/Command Prompt and run:
```powershell
cd C:\Users\ankur\Downloads\12\12
python api_with_auth.py
```

You should see:
```
============================================================
API Server Starting...
============================================================

Available API Keys:
  PfeQLfnlHgUKrdpRoB3E7kpGdOaKo-kkHXGeFg5Iyzw... - API Key

 * Running on http://0.0.0.0:5000
```

**⚠️ IMPORTANT: Keep this window open!** The server must stay running.

## 🧪 Step 3: Test the API

Open a **NEW terminal window** and run your test:

```powershell
cd C:\Users\ankur\OneDrive\Desktop\help_help2
node test-api-correct.js
```

Or use the corrected test file I created:
```powershell
cd C:\Users\ankur\Downloads\12\12
node test-api-correct.js
```

## 📋 Your API Key

```
PfeQLfnlHgUKrdpRoB3E7kpGdOaKo-kkHXGeFg5Iyzw
```

## ✅ Correct Data Format

Make sure your test data includes these fields:

```javascript
{
    equipment_type: 'CCTV Camera',
    manufacturer: 'Schneider',
    model: 'Schneider-120',
    facility_type: 'Academic',
    facility_location: 'Main Campus, Bangalore',
    floor_count: 1,
    total_area_sqm: 1200,
    incident_category: 'Network Outage',
    year: 2023,
    month: 10,
    day_of_week: 5,
    hour_of_day: 18,
    is_weekend: 0,
    season: 'Autumn',
    equipment_age_days: 1382.0,
    severity: 'Medium',
    downtime_hours: 22.3,
    cost_estimate: 25818.0,
    days_since_maintenance: -735.0,
    maintenance_cycle_days: 44
}
```

## 🔧 Troubleshooting

### Server won't start?
- Make sure you're in: `C:\Users\ankur\Downloads\12\12`
- Check if port 5000 is already in use
- Make sure all dependencies are installed: `pip install -r requirements.txt`

### "Connection refused" error?
- Make sure the API server is running (Step 2)
- Check the server window for errors

### "Invalid API key" error?
- Verify your API key matches: `PfeQLfnlHgUKrdpRoB3E7kpGdOaKo-kkHXGeFg5Iyzw`
- List all keys: `python api_key_manager.py list`

