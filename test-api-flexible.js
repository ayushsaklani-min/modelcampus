/**
 * Flexible Test Script for Time-to-Failure Prediction API
 * Adjust the data object below to match your needs
 */

const API_URL = 'http://localhost:5000/predict';
const API_KEY = 'PfeQLfnlHgUKrdpRoB3E7kpGdOaKo-kkHXGeFg5Iyzw'; // Your API key

// ============================================
// REQUIRED FIELDS (Minimum to make request)
// ============================================
const minimalData = {
    equipment_type: 'CCTV Camera',
    manufacturer: 'Schneider',
    facility_type: 'Academic',
    equipment_age_days: 1382.0,
    month: 10,
    year: 2023
};

// ============================================
// COMPLETE DATA (All available fields)
// ============================================
const completeData = {
    // Required fields
    equipment_type: 'CCTV Camera',
    manufacturer: 'Schneider',
    facility_type: 'Academic',
    equipment_age_days: 1382.0,
    month: 10,
    year: 2023,
    
    // Optional but recommended fields
    model: 'Schneider-120',
    facility_location: 'Main Campus, Bangalore',
    floor_count: 1,
    total_area_sqm: 1200,
    incident_category: 'Network Outage',
    day_of_week: 5,              // 0=Monday, 6=Sunday
    hour_of_day: 18,             // 0-23
    is_weekend: 0,                // 0 or 1
    season: 'Autumn',             // 'Autumn', 'Monsoon', 'Winter', 'Summer'
    severity: 'Medium',           // 'Low', 'Medium', 'High', 'Critical'
    downtime_hours: 22.3,
    cost_estimate: 25818.0,
    days_since_maintenance: -735.0,
    maintenance_cycle_days: 44
};

// ============================================
// YOUR CUSTOM DATA (Edit this section)
// ============================================
const yourData = {
    // REQUIRED - Must include these:
    equipment_type: 'CCTV Camera',      // Change to your equipment type
    manufacturer: 'Schneider',           // Change to your manufacturer
    facility_type: 'Academic',          // Change to your facility type
    equipment_age_days: 1382.0,         // Change to equipment age in days
    month: 10,                          // Current month (1-12)
    year: 2023,                         // Current year
    
    // OPTIONAL - Add more fields for better accuracy:
    // model: 'Your-Model-Number',
    // facility_location: 'Your Location',
    // floor_count: 1,
    // total_area_sqm: 1200,
    // incident_category: 'Network Outage',
    // day_of_week: 5,
    // hour_of_day: 18,
    // is_weekend: 0,
    // season: 'Autumn',
    // severity: 'Medium',
    // downtime_hours: 22.3,
    // cost_estimate: 25818.0,
    // days_since_maintenance: -735.0,
    // maintenance_cycle_days: 44
};

// ============================================
// TEST FUNCTION
// ============================================
async function testAPI(requestData, label = 'Test') {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`${label} - Testing Prediction API`);
    console.log('='.repeat(60));
    console.log(`API URL: ${API_URL}`);
    console.log(`API Key: ${API_KEY.substring(0, 20)}...\n`);
    console.log('Request Data:');
    console.log(JSON.stringify(requestData, null, 2));
    console.log('\n' + '-'.repeat(60) + '\n');

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

        if (response.ok && result.success) {
            console.log('✅ SUCCESS!');
            console.log('\nPrediction Results:');
            console.log(`  Days to Failure: ${result.prediction.days_to_failure} days`);
            console.log(`  Weeks to Failure: ${result.prediction.weeks_to_failure} weeks`);
            console.log(`  Months to Failure: ${result.prediction.months_to_failure} months`);
            console.log(`\nAPI Key Info: ${result.api_key_info.name} (Rate Limit: ${result.api_key_info.rate_limit})`);
            return result;
        } else {
            console.log('❌ ERROR');
            console.log(`Status: ${response.status}`);
            console.log('Response:', JSON.stringify(result, null, 2));
            
            if (result.error) {
                if (result.error.includes('Missing required fields')) {
                    console.log('\n💡 TIP: Make sure you include all required fields!');
                }
                if (result.error.includes('API key')) {
                    console.log('\n💡 TIP: Check your API key is correct!');
                }
            }
            return null;
        }
    } catch (error) {
        console.log('❌ CONNECTION ERROR');
        console.log(`Error: ${error.message}`);
        
        if (error.code === 'ECONNREFUSED' || error.message.includes('fetch failed')) {
            console.log('\n⚠️  The API server is not running!');
            console.log('Start it with:');
            console.log('  cd C:\\Users\\ankur\\Downloads\\12\\12');
            console.log('  python api_with_auth.py');
        }
        return null;
    }
}

// ============================================
// RUN TESTS
// ============================================
async function runTests() {
    console.log('🧪 Time-to-Failure Prediction API Tester\n');
    
    // Test 1: Minimal data (only required fields)
    await testAPI(minimalData, 'Test 1: Minimal Data');
    
    // Wait a bit between tests
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Test 2: Complete data (all fields)
    await testAPI(completeData, 'Test 2: Complete Data');
    
    // Wait a bit between tests
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Test 3: Your custom data
    await testAPI(yourData, 'Test 3: Your Custom Data');
    
    console.log('\n' + '='.repeat(60));
    console.log('✅ All tests completed!');
    console.log('='.repeat(60));
}

// Run the tests
runTests().catch(console.error);

