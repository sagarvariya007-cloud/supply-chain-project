import pandas as pd
import joblib, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/sales_data.csv", parse_dates=["Date"])

# Basic date features
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Sort data for lag features
df = df.sort_values(['Product', 'Region', 'Date'])

# Lag features
df['lag_1'] = df.groupby(['Product','Region'])['Sales'].shift(1)
df['lag_7'] = df.groupby(['Product','Region'])['Sales'].shift(7)

# 7-day moving average (FIXED version)
df['MA_7'] = (
    df.groupby(['Product','Region'])['Sales']
      .rolling(7, min_periods=1)
      .mean()
      .reset_index(level=[0,1], drop=True)
)

# Remove rows with NaN from lags
df.dropna(inplace=True)

# Encoding categorical features
le_prod = LabelEncoder()
df['Product_enc'] = le_prod.fit_transform(df['Product'])

le_reg = LabelEncoder()
df['Region_enc'] = le_reg.fit_transform(df['Region'])

# Feature list
FEATURES = [
    'Product_enc','Region_enc','Price','Promo','Inventory',
    'Day','Month','Year','DayOfWeek','lag_1','lag_7','MA_7'
]

X = df[FEATURES]
y = df['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ML pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    ))
])

# Train model
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

# Show accuracy
print("MAE:", mean_absolute_error(y_test, pred))

# Save model
joblib.dump(
    {
        'pipeline': pipe,
        'le_prod': le_prod,
        'le_reg': le_reg,
        'features': FEATURES
    },
    "model/demand_pipeline.pkl"
)

print("Saved model to model/demand_pipeline.pkl")
