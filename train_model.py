"""
Time-to-Failure Prediction Model Training Script
Replaces Vertex AI AutoML with local training using XGBoost and Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Time-to-Failure Prediction Model Training")
print("=" * 60)

# Load the dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv('vertex_ai_training_time_to_failure.csv')
print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# Check for missing values
print("\n[2/6] Data preprocessing...")
print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Separate features and target
target = 'days_to_next_failure'
X = df.drop(columns=[target])
y = df[target]

# Handle missing values in target
if y.isnull().sum() > 0:
    print(f"Removing {y.isnull().sum()} rows with missing target values")
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# Encode categorical variables
print("\n[3/6] Encoding categorical variables...")
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Handle missing values in features
print("\n[4/6] Handling missing values...")
for col in numerical_cols:
    if X_encoded[col].isnull().sum() > 0:
        X_encoded[col].fillna(X_encoded[col].median(), inplace=True)

for col in categorical_cols:
    if X_encoded[col].isnull().sum() > 0:
        X_encoded[col].fillna(0, inplace=True)

# Split the data
print("\n[5/6] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale numerical features (optional but can help some models)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train multiple models
print("\n[6/6] Training models...")
print("-" * 60)

models = {}
results = {}

# Model 1: Random Forest (similar to your Model 2)
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
models['RandomForest'] = rf_model

# Predictions
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

# Metrics
rf_mae_train = mean_absolute_error(y_train, rf_pred_train)
rf_mae_test = mean_absolute_error(y_test, rf_pred_test)
rf_rmse_train = np.sqrt(mean_squared_error(y_train, rf_pred_train))
rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_pred_test))
rf_r2_train = r2_score(y_train, rf_pred_train)
rf_r2_test = r2_score(y_test, rf_pred_test)

results['RandomForest'] = {
    'MAE_train': rf_mae_train,
    'MAE_test': rf_mae_test,
    'RMSE_train': rf_rmse_train,
    'RMSE_test': rf_rmse_test,
    'R2_train': rf_r2_train,
    'R2_test': rf_r2_test
}

print(f"  Train MAE: {rf_mae_train:.2f} days")
print(f"  Test MAE: {rf_mae_test:.2f} days")
print(f"  Train RMSE: {rf_rmse_train:.2f} days")
print(f"  Test RMSE: {rf_rmse_test:.2f} days")
print(f"  Train R²: {rf_r2_train:.4f}")
print(f"  Test R²: {rf_r2_test:.4f}")

# Model 2: XGBoost (similar to your Model 3)
print("\nTraining XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model

# Predictions
xgb_pred_train = xgb_model.predict(X_train)
xgb_pred_test = xgb_model.predict(X_test)

# Metrics
xgb_mae_train = mean_absolute_error(y_train, xgb_pred_train)
xgb_mae_test = mean_absolute_error(y_test, xgb_pred_test)
xgb_rmse_train = np.sqrt(mean_squared_error(y_train, xgb_pred_train))
xgb_rmse_test = np.sqrt(mean_squared_error(y_test, xgb_pred_test))
xgb_r2_train = r2_score(y_train, xgb_pred_train)
xgb_r2_test = r2_score(y_test, xgb_pred_test)

results['XGBoost'] = {
    'MAE_train': xgb_mae_train,
    'MAE_test': xgb_mae_test,
    'RMSE_train': xgb_rmse_train,
    'RMSE_test': xgb_rmse_test,
    'R2_train': xgb_r2_train,
    'R2_test': xgb_r2_test
}

print(f"  Train MAE: {xgb_mae_train:.2f} days")
print(f"  Test MAE: {xgb_mae_test:.2f} days")
print(f"  Train RMSE: {xgb_rmse_train:.2f} days")
print(f"  Test RMSE: {xgb_rmse_test:.2f} days")
print(f"  Train R²: {xgb_r2_train:.4f}")
print(f"  Test R²: {xgb_r2_test:.4f}")

# Model 3: XGBoost with scaled features
print("\nTraining XGBoost Regressor (with scaled features)...")
xgb_scaled_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_scaled_model.fit(X_train_scaled, y_train)
models['XGBoost_Scaled'] = xgb_scaled_model

# Predictions
xgb_scaled_pred_train = xgb_scaled_model.predict(X_train_scaled)
xgb_scaled_pred_test = xgb_scaled_model.predict(X_test_scaled)

# Metrics
xgb_scaled_mae_train = mean_absolute_error(y_train, xgb_scaled_pred_train)
xgb_scaled_mae_test = mean_absolute_error(y_test, xgb_scaled_pred_test)
xgb_scaled_rmse_train = np.sqrt(mean_squared_error(y_train, xgb_scaled_pred_train))
xgb_scaled_rmse_test = np.sqrt(mean_squared_error(y_test, xgb_scaled_pred_test))
xgb_scaled_r2_train = r2_score(y_train, xgb_scaled_pred_train)
xgb_scaled_r2_test = r2_score(y_test, xgb_scaled_pred_test)

results['XGBoost_Scaled'] = {
    'MAE_train': xgb_scaled_mae_train,
    'MAE_test': xgb_scaled_mae_test,
    'RMSE_train': xgb_scaled_rmse_train,
    'RMSE_test': xgb_scaled_rmse_test,
    'R2_train': xgb_scaled_r2_train,
    'R2_test': xgb_scaled_r2_test
}

print(f"  Train MAE: {xgb_scaled_mae_train:.2f} days")
print(f"  Test MAE: {xgb_scaled_mae_test:.2f} days")
print(f"  Train RMSE: {xgb_scaled_rmse_train:.2f} days")
print(f"  Test RMSE: {xgb_scaled_rmse_test:.2f} days")
print(f"  Train R²: {xgb_scaled_r2_train:.4f}")
print(f"  Test R²: {xgb_scaled_r2_test:.4f}")

# Select best model based on test R² score
print("\n" + "=" * 60)
print("Model Comparison Summary")
print("=" * 60)

best_model_name = max(results.keys(), key=lambda k: results[k]['R2_test'])
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"  Test R² Score: {results[best_model_name]['R2_test']:.4f}")
print(f"  Test MAE: {results[best_model_name]['MAE_test']:.2f} days")
print(f"  Test RMSE: {results[best_model_name]['RMSE_test']:.2f} days")

print("\nAll Models Performance:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Test R²: {metrics['R2_test']:.4f}")
    print(f"  Test MAE: {metrics['MAE_test']:.2f} days")
    print(f"  Test RMSE: {metrics['RMSE_test']:.2f} days")

# Save the best model and preprocessing objects
print("\n" + "=" * 60)
print("Saving model and preprocessing objects...")
print("=" * 60)

# Determine which preprocessing to save
if 'Scaled' in best_model_name:
    model_to_save = xgb_scaled_model
    use_scaler = True
else:
    model_to_save = best_model
    use_scaler = False

# Save model
joblib.dump(model_to_save, 'best_model.pkl')
print("✓ Saved: best_model.pkl")

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✓ Saved: label_encoders.pkl")

# Save scaler if used
if use_scaler:
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Saved: scaler.pkl")

# Save feature information
feature_info = {
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'feature_order': X_encoded.columns.tolist(),
    'use_scaler': use_scaler,
    'model_type': best_model_name
}
joblib.dump(feature_info, 'feature_info.pkl')
print("✓ Saved: feature_info.pkl")

# Save results summary
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv', index=True)
print("✓ Saved: model_results.csv")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"\nBest model ({best_model_name}) saved and ready for deployment.")
print("\nFiles created:")
print("  - best_model.pkl (trained model)")
print("  - label_encoders.pkl (categorical encoders)")
print("  - scaler.pkl (feature scaler, if used)")
print("  - feature_info.pkl (feature metadata)")
print("  - model_results.csv (performance metrics)")

