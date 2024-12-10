import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import warnings

# --- Loading datasets ---
print("Loading datasets...")
calendar_file = 'data/calendar.csv.gz'
listings_file = 'data/listings.csv.gz'
neighborhood_file = 'data/neighbourhoods.csv'

calendar_df = pd.read_csv(calendar_file, compression='gzip', low_memory=False)
listings_df = pd.read_csv(listings_file, compression='gzip')
neighborhoods_df = pd.read_csv(neighborhood_file)

# --- Cleaning and preprocessing data ---
print("\nCleaning and preprocessing data...")
listings_df = listings_df.dropna(subset=['price'])
listings_df['price'] = listings_df['price'].replace('[\$,]', '', regex=True).astype(float)
listings_df['log_price'] = np.log1p(listings_df['price'])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
for col in ['beds', 'bedrooms']:
    if col in listings_df.columns:
        listings_df[col] = imputer.fit_transform(listings_df[[col]])

# Feature: Beds per bedroom
listings_df['beds_per_bedroom'] = listings_df['beds'] / listings_df['bedrooms'].replace(0, np.nan)

# Drop constant or near-constant features
low_variance_cols = [col for col in listings_df.columns if listings_df[col].nunique() <= 1]
listings_df = listings_df.drop(columns=low_variance_cols)

# --- Neighborhood Data Integration ---
print("\nIntegrating neighborhood data...")
if 'neighbourhood' in listings_df.columns:
    listings_df = listings_df.merge(neighborhoods_df, on='neighbourhood', how='left', suffixes=('', '_neighborhood'))

# --- Encoding Categorical Features ---
categorical_features = ['room_type', 'neighbourhood_group'] if 'neighbourhood_group' in listings_df.columns else ['room_type']
for feature in categorical_features:
    if feature in listings_df.columns:
        listings_df[feature] = listings_df[feature].astype('category').cat.codes

# --- Numerical Features ---
numerical_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                      'calculated_host_listings_count', 'availability_365', 'beds_per_bedroom']
numerical_features = [col for col in numerical_features if col in listings_df.columns]

# Combine all features
features = categorical_features + numerical_features
if not features:
    raise ValueError("No valid features available for training.")

print(f"Final set of features: {features}")

# --- Splitting and Scaling Data ---
X = listings_df[features]
y = listings_df['log_price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_features])
X_scaled = np.hstack([X_scaled, X[categorical_features].values])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- LightGBM Model Training ---
print("\nTraining LightGBM model...")
warnings.filterwarnings('ignore') # suppress general warnings

lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42, verbose=-1)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [6, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'min_data_in_leaf': [20, 50],
    'feature_fraction': [0.8, 1.0],
    'num_leaves': [15, 31]
}

grid_search = GridSearchCV(lgb_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# --- Evaluation ---
print("\nEvaluating model...")
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = model.score(X_test, y_test)

print(f"Validation MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

# --- Feature Importance Visualization ---
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
plt.title("Feature Importance")
plt.savefig("output/visualizations/lightgbm_feature_importance.png")
plt.close()
print("Feature importance visualization saved.")
