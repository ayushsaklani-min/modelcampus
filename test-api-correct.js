/**
 * Correct Test File for Time-to-Failure Prediction API
 * This uses the correct data format that the model expects
 */

const API_URL = 'http://localhost:5000/predict';
const API_KEY = 'PfeQLfnlHgUKrdpRoB3E7kpGdOaKo-kkHXGeFg5Iyzw'; // Your generated key

async function testAPI() {
    console.log('Testing Prediction API...\n');
    console.log(`API URL: ${API_URL}`);
    console.log(`API Key: ${API_KEY.substring(0, 20)}...\n`);

    // CORRECT DATA FORMAT - Equipment failure prediction
    const requestData = {
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
    };

    console.log('Request Data:', JSON.stringify(requestData, null, 2));
    console.log('\n---\n');

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();

        if (response.ok) {
            console.log('✅ Success!');
            console.log('\nPrediction Result:');
            console.log(`  Days to Failure: ${result.prediction.days_to_failure} days`);
            console.log(`  Weeks to Failure: ${result.prediction.weeks_to_failure} weeks`);
            console.log(`  Months to Failure: ${result.prediction.months_to_failure} months`);
        } else {
            console.log('❌ Error:', result.error || 'Unknown error');
            console.log('Full response:', JSON.stringify(result, null, 2));
        }
    } catch (error) {
        console.log('❌ Error:', error.message);
        console.log('\nFull error:', error);
        
        if (error.code === 'ECONNREFUSED') {
            console.log('\n⚠️  The API server is not running!');
            console.log('Start it with: python api_with_auth.py');
            console.log('Make sure you are in the directory: C:\\Users\\ankur\\Downloads\\12\\12');
        }
    }
}

// Run the test
testAPI();

