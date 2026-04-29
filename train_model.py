import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/Housing_price.csv")

# ----------- Encode yes/no columns -----------
yes_no_cols = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# ----------- One-Hot Encode furnishingstatus (FIXED) -----------
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

# ----------- Features & Target -----------
X = df.drop("price", axis=1)
y = df["price"]

# ----------- Save column structure (VERY IMPORTANT) -----------
columns = X.columns.tolist()

# ----------- Scaling (optional for RF, but kept for consistency) -----------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ----------- Train/Test Split -----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------- Train Model -----------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)

# ----------- Evaluation (INTERVIEW MUST) -----------
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("📊 Model Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# ----------- Save Everything -----------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/house_price_model.pkl")
joblib.dump(scaler, "model/scaler.save")
joblib.dump(columns, "model/columns.pkl")  # 🔥 KEY FILE

print("✅ Model trained & saved successfully!")