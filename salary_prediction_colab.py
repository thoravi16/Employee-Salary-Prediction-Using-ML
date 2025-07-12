#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# load dataset
df = pd.read_csv("adult 3.csv", on_bad_lines='skip')
df.head()

# Clean missing values
df = df.replace(" ?", np.nan)
df.dropna(inplace=True)

# Convert target label to binary (classification)
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

# Feature matrix (X) and target (y)
X = df.drop('income', axis=1)
y = df['income']

# Split categorical and numeric columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include='int64').columns.tolist()

# Define preprocessing transformers
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Define base models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Ensemble model
ensemble = VotingRegressor(estimators=[
    ('rf', rf),
    ('gb', gb)
])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ensemble)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

#save the models
import joblib
joblib.dump(pipeline, 'salary_model.pkl')

# Extract feature importances and plot top 20 (Random Forest only)
rf_model = pipeline.named_steps['regressor'].estimators_[0]
importances = rf_model.feature_importances_

# Get encoded categorical + numerical feature names
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot']
encoded_cols = ohe.get_feature_names_out(categorical_cols)
feature_names = np.array(list(encoded_cols) + numerical_cols)

# Get top 20 feature indices
indices = np.argsort(importances)[-20:]
top_features = feature_names[indices]
top_importances = importances[indices]

# Plot top 20
plt.figure(figsize=(10, 8))
plt.barh(top_features, top_importances)
plt.xlabel("Importance")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

