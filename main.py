import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load datasets
train_file = "data/newzealandlistings_train.csv"
test_file = "data/newzealandlistings_test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Data Cleaning
print("\nCleaning data...")
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna(subset=features + [target])

    # Convert price to numeric (remove $ sign if present)
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

    # Remove outliers in price (e.g., prices below $10 or above $5000)
    df = df[(df['price'] > 10) & (df['price'] < 5000)]

    # Create new features (e.g., beds per bedroom)
    df['beds_per_bedroom'] = df['beds'] / df['bedrooms']
    df['beds_per_bedroom'].fillna(0, inplace=True)  # Handle divisions by zero
    
    return df

# Features and target variable
features = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
target = 'price'

# Clean train and test datasets
train_df = clean_data(train_df)
test_df = clean_data(test_df)

# Log-transform the price to stabilize variance
train_df['price'] = np.log1p(train_df['price'])
test_df['price'] = np.log1p(test_df['price'])

# Separate features and target
X_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df[target], test_size=0.2, random_state=42)
X_test = test_df[features]
y_test = test_df[target]

# Hyperparameter tuning
print("\nOptimizing model...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
model.fit(X_train, y_train)

# Best model
print(f"Best parameters: {model.best_params_}")
best_model = model.best_estimator_

# Evaluate model
print("\nEvaluating model...")
val_predictions = best_model.predict(X_val)
test_predictions = best_model.predict(X_test)

# Metrics
val_mse = mean_squared_error(y_val, val_predictions)
val_mae = mean_absolute_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)

test_mse = mean_squared_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"Validation MSE: {val_mse:.2f}, MAE: {val_mae:.2f}, R2: {val_r2:.2f}")
print(f"Test MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.2f}")

# Save a visualization of feature importance
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
for i, value in enumerate(feature_importance.values):
    plt.text(value, i, f"{value:.2f}", va='center')
plt.tight_layout()
plt.savefig("output/visualizations/feature_importance.png")
plt.close()
print("Feature importance visualization saved.")

# Residual Plot
plt.figure(figsize=(10, 6))
residuals = np.expm1(y_test) - np.expm1(test_predictions)
sns.histplot(residuals, kde=True, color="blue", bins=30)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.savefig("output/visualizations/residuals_distribution.png")
plt.close()
