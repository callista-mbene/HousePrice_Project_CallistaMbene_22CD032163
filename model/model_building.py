# House Price Prediction Model Development

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('train.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")

# 2. Feature Selection - Select 6 features from the 9 recommended
# Selected features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood
selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood']
target = 'SalePrice'

# Create a subset with selected features and target
df_subset = df[selected_features + [target]].copy()
print(f"\nSelected features dataset shape: {df_subset.shape}")

# 3. Data Preprocessing

# a. Handling missing values
print("\nMissing values before handling:")
print(df_subset.isnull().sum())

# Fill missing values
df_subset['TotalBsmtSF'].fillna(0, inplace=True)
df_subset['GarageCars'].fillna(0, inplace=True)

print("\nMissing values after handling:")
print(df_subset.isnull().sum())

# b. Encoding categorical variables (Neighborhood)
le = LabelEncoder()
df_subset['Neighborhood_Encoded'] = le.fit_transform(df_subset['Neighborhood'])

# Save the label encoder for later use
joblib.dump(le, 'model/label_encoder.pkl')
print("\nLabel Encoder saved!")

# c. Prepare features and target
X = df_subset[['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood_Encoded']]
y = df_subset[target]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 5. Train the model - Random Forest Regressor
print("\nTraining Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20, min_samples_split=5)
model.fit(X_train, y_train)
print("Model training completed!")

# 6. Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 7. Evaluate the model
print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)

# Training metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_pred_train)

print("\nTraining Set Metrics:")
print(f"MAE (Mean Absolute Error): ${train_mae:,.2f}")
print(f"MSE (Mean Squared Error): ${train_mse:,.2f}")
print(f"RMSE (Root Mean Squared Error): ${train_rmse:,.2f}")
print(f"R² Score: {train_r2:.4f}")

# Test metrics
test_mae = mean_absolute_error(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred_test)

print("\nTest Set Metrics:")
print(f"MAE (Mean Absolute Error): ${test_mae:,.2f}")
print(f"MSE (Mean Squared Error): ${test_mse:,.2f}")
print(f"RMSE (Root Mean Squared Error): ${test_rmse:,.2f}")
print(f"R² Score: {test_r2:.4f}")

# 8. Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# 9. Save the trained model
joblib.dump(model, 'model/house_price_model.pkl')
print("\nModel saved successfully to 'model/house_price_model.pkl'")

# 10. Test loading the saved model
print("\nTesting model reload...")
loaded_model = joblib.load('model/house_price_model.pkl')
loaded_le = joblib.load('model/label_encoder.pkl')

# Make a test prediction
test_input = [[7, 1500, 1000, 2, 2005, 0]]  # Example input
test_prediction = loaded_model.predict(test_input)
print(f"Test prediction with loaded model: ${test_prediction[0]:,.2f}")
print("\nModel successfully reloaded and tested!")

print("\n" + "="*50)
print("MODEL DEVELOPMENT COMPLETED!")
print("="*50)