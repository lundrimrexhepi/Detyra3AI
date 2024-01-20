import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Load the dataset
file_path = "snow.csv"
df = pd.read_csv(file_path)

df = df.sample(10000)

# Assuming your target variable is 'Snow' and features are 'Latitude' and 'Longitude'
X = df[['Latitude', 'Longitude']]
y = df['Snow']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM regression model
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_mse = mean_squared_error(y_test, lgb_pred)

# Train XGBoost regression model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred)

# Train CatBoost regression model
catboost_model = CatBoostRegressor()
catboost_model.fit(X_train, y_train)
catboost_pred = catboost_model.predict(X_test)
catboost_mse = mean_squared_error(y_test, catboost_pred)

print("CatBoost Mean Squared Error:", catboost_mse)
print("LightGBM Mean Squared Error:", lgb_mse)
print("XGBoost Mean Squared Error:", xgb_mse)
