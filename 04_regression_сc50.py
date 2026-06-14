import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv('cleaned_molecular_data.csv')

# подготовка признаков (X) и таргета (y)
targets_to_drop = ['IC50, mM', 'CC50, mM', 'SI', 'IC50, mM_log', 'CC50, mM_log', 'SI_log']
X = df.drop(columns=targets_to_drop)
y = df['CC50, mM_log']

# разделение на train и test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# масштабирование признаков 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- МОДЕЛЬ 1: RIDGE REGRESSION ---
print("--- Обучение Ridge Regression ---")
ridge = Ridge()
param_grid_ridge = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]} 
grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
grid_ridge.fit(X_train_scaled, y_train)

best_ridge = grid_ridge.best_estimator_
y_pred_ridge_log = best_ridge.predict(X_test_scaled)

# возвращение предсказания из логарифма в исходную шкалу mM
y_pred_ridge = np.expm1(y_pred_ridge_log)
y_test_real = np.expm1(y_test)

print(f"Лучшие параметры Ridge: {grid_ridge.best_params_}")
print(f"R2 (на логарифмах): {r2_score(y_test, y_pred_ridge_log):.4f}")
print(f"RMSE (в реальных mM): {np.sqrt(mean_squared_error(y_test_real, y_pred_ridge)):.4f}")


# --- МОДЕЛЬ 2: RANDOM FOREST ---
print("\n--- Обучение Random Forest ---")
rf = RandomForestRegressor(random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200], 
    'max_depth': [None, 5, 10],  
    'min_samples_split': [2, 5] 
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf_log = best_rf.predict(X_test_scaled)

# возврат предсказания в исходную шкалу 
y_pred_rf = np.expm1(y_pred_rf_log)

print(f"Лучшие параметры Random Forest: {grid_rf.best_params_}")
print(f"R2 (на логарифмах): {r2_score(y_test, y_pred_rf_log):.4f}")
print(f"RMSE (в реальных mM): {np.sqrt(mean_squared_error(y_test_real, y_pred_rf)):.4f}")

# --- МОДЕЛЬ 3: XGBoost ---
print("\n--- Обучение XGBoost ---")
xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1], 
    'max_depth': [3, 5, 7] 
}

grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train)

# лучшая модель и  предсказания
best_xgb = grid_xgb.best_estimator_
y_pred_xgb_log = best_xgb.predict(X_test_scaled)

# возврат предсказания в исходную шкалу 
y_pred_xgb = np.expm1(y_pred_xgb_log)

print(f"Лучшие параметры XGBoost: {grid_xgb.best_params_}")
print(f"R2 (на логарифмах): {r2_score(y_test, y_pred_xgb_log):.4f}")
print(f"RMSE (в реальных mM): {np.sqrt(mean_squared_error(y_test_real, y_pred_xgb)):.4f}")