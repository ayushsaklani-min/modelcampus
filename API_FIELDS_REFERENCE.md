# API Fields Reference

## Required Fields (Minimum)

The API validates these fields are present:

```javascript
{
    "equipment_type": "CCTV Camera",      // REQUIRED
    "manufacturer": "Schneider",          // REQUIRED
    "facility_type": "Academic",          // REQUIRED
    "equipment_age_days": 1382.0,         // REQUIRED (number)
    "month": 10,                          // REQUIRED (1-12)
    "year": 2023                          // REQUIRED
}
```

## All Available Fields (Complete Format)

Based on the training data, here are ALL fields the model can use:

```javascript
{
    // Equipment Information
    "equipment_type": "CCTV Camera",           // e.g., "CCTV Camera", "Fire Alarm", "UPS System", "Elevator", "Generator", "Water Pump", "Electrical Panel"
    "manufacturer": "Schneider",              // e.g., "Schneider", "Emerson", "Crompton", "Siemens", "Carrier", "Cummins", "Bosch"
    "model": "Schneider-120",                 // Model number/name
    
    // Facility Information
    "facility_type": "Academic",              // e.g., "Academic", "Hostel", "Administrative", "Parking", "Library", "Healthcare", "Cafeteria"
    "facility_location": "Main Campus, Bangalore",  // Location string
    "floor_count": 1,                         // Number of floors (integer)
    "total_area_sqm": 1200,                   // Total area in square meters (number)
    
    // Incident Information
    "incident_category": "Network Outage",    // e.g., "Network Outage", "IoT Sensor Fault", "Plumbing Issues", "Overheating", "Fire/Smoke Alert", "Security Event", "Power Fluctuation"
    "severity": "Medium",                     // e.g., "Low", "Medium", "High", "Critical"
    "downtime_hours": 22.3,                   // Hours of downtime (number)
    "cost_estimate": 25818.0,                 // Estimated cost (number)
    
    // Time Information
    "year": 2023,                             // Year (integer)
    "month": 10,                              // Month 1-12 (integer)
    "day_of_week": 5,                         // Day of week 0-6 (0=Monday, 6=Sunday) (integer)
    "hour_of_day": 18,                        // Hour 0-23 (integer)
    "is_weekend": 0,                          // 0 or 1 (integer)
    "season": "Autumn",                       // e.g., "Autumn", "Monsoon", "Winter", "Summer"
    
    // Equipment Age & Maintenance
    "equipment_age_days": 1382.0,             // Age in days (number)
    "days_since_maintenance": -735.0,          // Days since last maintenance (can be negative) (number)
    "maintenance_cycle_days": 44              // Maintenance cycle in days (number)
}
```

## Field Types

- **Strings**: `equipment_type`, `manufacturer`, `model`, `facility_type`, `facility_location`, `incident_category`, `severity`, `season`
- **Integers**: `year`, `month`, `day_of_week`, `hour_of_day`, `is_weekend`, `floor_count`
- **Numbers (Float)**: `equipment_age_days`, `downtime_hours`, `cost_estimate`, `days_since_maintenance`, `maintenance_cycle_days`, `total_area_sqm`

## Example Valid Values

### Equipment Types:
- CCTV Camera
- Fire Alarm
- UPS System
- Elevator
- Generator
- Water Pump
- Electrical Panel

### Manufacturers:
- Schneider
- Emerson
- Crompton
- Siemens
- Carrier
- Cummins
- Bosch

### Facility Types:
- Academic
- Hostel
- Administrative
- Parking
- Library
- Healthcare
- Cafeteria

### Severity Levels:
- Low
- Medium
- High
- Critical

### Seasons:
- Autumn
- Monsoon
- Winter
- Summer

### Incident Categories:
- Network Outage
- IoT Sensor Fault
- Plumbing Issues
- Overheating
- Fire/Smoke Alert
- Security Event
- Power Fluctuation

## Minimal Working Example

```javascript
{
    "equipment_type": "CCTV Camera",
    "manufacturer": "Schneider",
    "facility_type": "Academic",
    "equipment_age_days": 1382.0,
    "month": 10,
    "year": 2023
}
```

## Complete Example

```javascript
{
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
}
```

## Notes

- **Missing fields**: If you don't provide optional fields, the model will use default values (usually 0 or median)
- **Unknown categories**: If you use a category value not seen during training, it will be encoded as 0
- **Negative values**: `days_since_maintenance` can be negative (meaning maintenance is scheduled in the future)

