import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

# --- Loading datasets ---
print("Loading datasets...")
calendar_df = pd.read_csv('data/calendar.csv.gz', compression='gzip', low_memory=False)
listings_df = pd.read_csv('data/listings.csv.gz', compression='gzip')
neighborhoods_df = pd.read_csv('data/neighbourhoods.csv')

# --- Cleaning & Preprocessing ---
print("\nCleaning and preprocessing data...")
listings_df = listings_df.dropna(subset=['price'])
listings_df['price'] = listings_df['price'].replace('[\$,]', '', regex=True).astype(float)
listings_df['log_price'] = np.log1p(listings_df['price'])

# Impute 'beds' & 'bedrooms'
imputer = SimpleImputer(strategy='median')
for col in ['beds', 'bedrooms']:
    if col in listings_df.columns:
        listings_df[col] = imputer.fit_transform(listings_df[[col]])

# Feature: Beds per bedroom
listings_df['beds_per_bedroom'] = listings_df['beds'] / listings_df['bedrooms'].replace(0, np.nan)

# --- Neighborhood Integration ---
print("\nIntegrating neighborhood data...")
if 'neighbourhood' in listings_df.columns:
    listings_df = listings_df.merge(neighborhoods_df, on='neighbourhood', how='left', suffixes=('', '_neighborhood'))

# --- Encoding Categorical Features ---
categorical_features = [col for col in ['room_type', 'neighbourhood_group'] if col in listings_df.columns]
listings_df[categorical_features] = listings_df[categorical_features].astype('category').apply(lambda x: x.cat.codes)

# --- Numerical Features ---
numerical_features = [
    'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'calculated_host_listings_count', 'availability_365', 'beds_per_bedroom'
]
numerical_features = [col for col in numerical_features if col in listings_df.columns]

# Final Feature Set
features = categorical_features + numerical_features
if not features:
    raise ValueError("No valid features available for training.")
print(f"Final set of features: {features}")

# --- Splitting Data ---
X = listings_df[features]
y = listings_df['log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training with RandomizedSearchCV ---
print("\nTuning and training model...")
param_dist = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=15,  # Limit iterations for speed
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
model = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

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
sns.barplot(
    x=feature_importance.values, 
    y=feature_importance.index, 
    hue=feature_importance.index, 
    dodge=False, 
    palette="viridis", 
    # legend=False
)

plt.title("Feature Importance")
plt.savefig("output/visualizations/randomForest_feature_importance.png")
plt.close()
print("Feature importance visualization saved.")
