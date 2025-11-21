"""
Fast Fine-Tuned Time-to-Failure Prediction Model Training Script
Optimized for speed while still improving model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import warnings
from scipy.stats import randint, uniform
warnings.filterwarnings('ignore')

print("=" * 70)
print("Fast Fine-Tuned Time-to-Failure Prediction Model Training")
print("=" * 70)

# Load the dataset
print("\n[1/7] Loading dataset...")
df = pd.read_csv('vertex_ai_training_time_to_failure.csv')
print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# Check for missing values
print("\n[2/7] Data preprocessing and feature engineering...")
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

# Feature Engineering
print("\n[3/7] Creating additional features...")
X_fe = X.copy()

# Create interaction features
if 'equipment_age_days' in X_fe.columns and 'days_since_maintenance' in X_fe.columns:
    X_fe['maintenance_ratio'] = X_fe['days_since_maintenance'] / (X_fe['equipment_age_days'] + 1)
    X_fe['age_maintenance_interaction'] = X_fe['equipment_age_days'] * X_fe['days_since_maintenance']

# Create time-based features
if 'month' in X_fe.columns:
    X_fe['is_peak_season'] = X_fe['month'].isin([6, 7, 8, 12, 1]).astype(int)

if 'hour_of_day' in X_fe.columns:
    X_fe['is_business_hours'] = ((X_fe['hour_of_day'] >= 9) & (X_fe['hour_of_day'] <= 17)).astype(int)

# Create cost efficiency features
if 'cost_estimate' in X_fe.columns and 'downtime_hours' in X_fe.columns:
    X_fe['cost_per_hour'] = X_fe['cost_estimate'] / (X_fe['downtime_hours'] + 1)

# Facility size features
if 'total_area_sqm' in X_fe.columns and 'floor_count' in X_fe.columns:
    X_fe['area_per_floor'] = X_fe['total_area_sqm'] / (X_fe['floor_count'] + 1)

# Equipment age categories
if 'equipment_age_days' in X_fe.columns:
    X_fe['equipment_age_years'] = X_fe['equipment_age_days'] / 365.25
    X_fe['is_old_equipment'] = (X_fe['equipment_age_days'] > 2000).astype(int)
    X_fe['is_new_equipment'] = (X_fe['equipment_age_days'] < 500).astype(int)

# Maintenance frequency features
if 'maintenance_cycle_days' in X_fe.columns:
    X_fe['maintenance_frequency'] = 365.25 / (X_fe['maintenance_cycle_days'] + 1)
    X_fe['is_high_maintenance'] = (X_fe['maintenance_cycle_days'] < 90).astype(int)

print(f"Original features: {len(X.columns)}")
print(f"Features after engineering: {len(X_fe.columns)}")
print(f"New features added: {len(X_fe.columns) - len(X.columns)}")

# Identify categorical and numerical columns
categorical_cols = X_fe.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_fe.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# Encode categorical variables
print("\n[4/7] Encoding categorical variables...")
label_encoders = {}
X_encoded = X_fe.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_fe[col].astype(str))
    label_encoders[col] = le

# Handle missing values in features
print("\n[5/7] Handling missing values...")
for col in numerical_cols:
    if X_encoded[col].isnull().sum() > 0:
        median_val = X_encoded[col].median()
        X_encoded[col].fillna(median_val, inplace=True)

for col in categorical_cols:
    if X_encoded[col].isnull().sum() > 0:
        X_encoded[col].fillna(0, inplace=True)

# Handle infinite values
X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
for col in numerical_cols:
    if X_encoded[col].isnull().sum() > 0:
        X_encoded[col].fillna(X_encoded[col].median(), inplace=True)

# Split the data
print("\n[6/7] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train and fine-tune models (FAST VERSION - fewer iterations)
print("\n[7/7] Training and fine-tuning models (fast mode)...")
print("=" * 70)

models = {}
results = {}
best_params = {}

# Use a smaller sample for hyperparameter tuning to speed up
tune_sample_size = min(10000, len(X_train))
X_tune = X_train.iloc[:tune_sample_size]
y_tune = y_train.iloc[:tune_sample_size]
print(f"Using {tune_sample_size} samples for hyperparameter tuning (for speed)")

# Model 1: Fine-tuned Random Forest (FAST - 10 iterations, 3 CV folds)
print("\n" + "-" * 70)
print("Fine-tuning Random Forest Regressor (fast mode)...")
print("-" * 70)

rf_param_dist = {
    'n_estimators': [200, 300, 400],
    'max_depth': [20, 25, 30, None],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_base, rf_param_dist, n_iter=10, cv=3,  # Reduced from 30 iterations and 5 CV
    scoring='neg_mean_absolute_error', n_jobs=-1, 
    random_state=42, verbose=1
)

print("Running hyperparameter search...")
rf_search.fit(X_tune, y_tune)
best_params['RandomForest'] = rf_search.best_params_

# Train final model on full dataset with best params
rf_model = RandomForestRegressor(**rf_search.best_params_, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
models['RandomForest_Tuned'] = rf_model

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

# Quick cross-validation (3 folds)
rf_cv_scores = cross_val_score(rf_model, X_train.iloc[:20000], y_train.iloc[:20000], 
                               cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_cv_mae = -rf_cv_scores.mean()

results['RandomForest_Tuned'] = {
    'MAE_train': rf_mae_train,
    'MAE_test': rf_mae_test,
    'RMSE_train': rf_rmse_train,
    'RMSE_test': rf_rmse_test,
    'R2_train': rf_r2_train,
    'R2_test': rf_r2_test,
    'CV_MAE': rf_cv_mae
}

print(f"\nBest Parameters: {rf_search.best_params_}")
print(f"  Train MAE: {rf_mae_train:.2f} days")
print(f"  Test MAE: {rf_mae_test:.2f} days")
print(f"  CV MAE: {rf_cv_mae:.2f} days")
print(f"  Train RMSE: {rf_rmse_train:.2f} days")
print(f"  Test RMSE: {rf_rmse_test:.2f} days")
print(f"  Train R²: {rf_r2_train:.4f}")
print(f"  Test R²: {rf_r2_test:.4f}")

# Model 2: Fine-tuned XGBoost (FAST - 15 iterations, 3 CV folds)
print("\n" + "-" * 70)
print("Fine-tuning XGBoost Regressor (fast mode)...")
print("-" * 70)

xgb_param_dist = {
    'n_estimators': [200, 300, 400],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
xgb_search = RandomizedSearchCV(
    xgb_base, xgb_param_dist, n_iter=15, cv=3,  # Reduced from 40 iterations and 5 CV
    scoring='neg_mean_absolute_error', n_jobs=-1,
    random_state=42, verbose=1
)

print("Running hyperparameter search...")
xgb_search.fit(X_tune, y_tune)
best_params['XGBoost'] = xgb_search.best_params_

# Train final model on full dataset with best params
xgb_model = xgb.XGBRegressor(**xgb_search.best_params_, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
models['XGBoost_Tuned'] = xgb_model

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

# Quick cross-validation
xgb_cv_scores = cross_val_score(xgb_model, X_train.iloc[:20000], y_train.iloc[:20000], 
                                cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
xgb_cv_mae = -xgb_cv_scores.mean()

results['XGBoost_Tuned'] = {
    'MAE_train': xgb_mae_train,
    'MAE_test': xgb_mae_test,
    'RMSE_train': xgb_rmse_train,
    'RMSE_test': xgb_rmse_test,
    'R2_train': xgb_r2_train,
    'R2_test': xgb_r2_test,
    'CV_MAE': xgb_cv_mae
}

print(f"\nBest Parameters: {xgb_search.best_params_}")
print(f"  Train MAE: {xgb_mae_train:.2f} days")
print(f"  Test MAE: {xgb_mae_test:.2f} days")
print(f"  CV MAE: {xgb_cv_mae:.2f} days")
print(f"  Train RMSE: {xgb_rmse_train:.2f} days")
print(f"  Test RMSE: {xgb_rmse_test:.2f} days")
print(f"  Train R²: {xgb_r2_train:.4f}")
print(f"  Test R²: {xgb_r2_test:.4f}")

# Model 3: Improved baseline (no tuning, but with better default params)
print("\n" + "-" * 70)
print("Training Improved XGBoost (optimized defaults)...")
print("-" * 70)

xgb_improved = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1
)
xgb_improved.fit(X_train, y_train)
models['XGBoost_Improved'] = xgb_improved

# Predictions
xgb_imp_pred_train = xgb_improved.predict(X_train)
xgb_imp_pred_test = xgb_improved.predict(X_test)

# Metrics
xgb_imp_mae_train = mean_absolute_error(y_train, xgb_imp_pred_train)
xgb_imp_mae_test = mean_absolute_error(y_test, xgb_imp_pred_test)
xgb_imp_rmse_train = np.sqrt(mean_squared_error(y_train, xgb_imp_pred_train))
xgb_imp_rmse_test = np.sqrt(mean_squared_error(y_test, xgb_imp_pred_test))
xgb_imp_r2_train = r2_score(y_train, xgb_imp_pred_train)
xgb_imp_r2_test = r2_score(y_test, xgb_imp_pred_test)

xgb_imp_cv_scores = cross_val_score(xgb_improved, X_train.iloc[:20000], y_train.iloc[:20000], 
                                    cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
xgb_imp_cv_mae = -xgb_imp_cv_scores.mean()

results['XGBoost_Improved'] = {
    'MAE_train': xgb_imp_mae_train,
    'MAE_test': xgb_imp_mae_test,
    'RMSE_train': xgb_imp_rmse_train,
    'RMSE_test': xgb_imp_rmse_test,
    'R2_train': xgb_imp_r2_train,
    'R2_test': xgb_imp_r2_test,
    'CV_MAE': xgb_imp_cv_mae
}

print(f"  Train MAE: {xgb_imp_mae_train:.2f} days")
print(f"  Test MAE: {xgb_imp_mae_test:.2f} days")
print(f"  CV MAE: {xgb_imp_cv_mae:.2f} days")
print(f"  Train RMSE: {xgb_imp_rmse_train:.2f} days")
print(f"  Test RMSE: {xgb_imp_rmse_test:.2f} days")
print(f"  Train R²: {xgb_imp_r2_train:.4f}")
print(f"  Test R²: {xgb_imp_r2_test:.4f}")

# Select best model
print("\n" + "=" * 70)
print("Model Comparison Summary")
print("=" * 70)

best_model_name = max(results.keys(), key=lambda k: results[k]['R2_test'])
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"  Test R² Score: {results[best_model_name]['R2_test']:.4f}")
print(f"  Test MAE: {results[best_model_name]['MAE_test']:.2f} days")
print(f"  Test RMSE: {results[best_model_name]['RMSE_test']:.2f} days")
print(f"  CV MAE: {results[best_model_name]['CV_MAE']:.2f} days")

print("\nAll Models Performance:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Test R²: {metrics['R2_test']:.4f}")
    print(f"  Test MAE: {metrics['MAE_test']:.2f} days")
    print(f"  Test RMSE: {metrics['RMSE_test']:.2f} days")
    print(f"  CV MAE: {metrics['CV_MAE']:.2f} days")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 70)
    print("Top 15 Most Important Features")
    print("=" * 70)
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# Save the best model and preprocessing objects
print("\n" + "=" * 70)
print("Saving model and preprocessing objects...")
print("=" * 70)

# Save model
joblib.dump(best_model, 'best_model.pkl')
print("✓ Saved: best_model.pkl")

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✓ Saved: label_encoders.pkl")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("✓ Saved: scaler.pkl")

# Save feature information
feature_info = {
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'feature_order': X_encoded.columns.tolist(),
    'use_scaler': True,
    'model_type': best_model_name,
    'best_params': best_params.get(best_model_name.split('_')[0], None)
}
joblib.dump(feature_info, 'feature_info.pkl')
print("✓ Saved: feature_info.pkl")

# Save results summary
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results_finetuned.csv', index=True)
print("✓ Saved: model_results_finetuned.csv")

# Save best parameters
import json
with open('best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print("✓ Saved: best_hyperparameters.json")

print("\n" + "=" * 70)
print("Fine-Tuning Complete!")
print("=" * 70)
print(f"\nBest model ({best_model_name}) saved and ready for deployment.")
print("\nFiles created:")
print("  - best_model.pkl (fine-tuned model)")
print("  - label_encoders.pkl (categorical encoders)")
print("  - scaler.pkl (feature scaler)")
print("  - feature_info.pkl (feature metadata)")
print("  - model_results_finetuned.csv (performance metrics)")
print("  - best_hyperparameters.json (optimal hyperparameters)")

