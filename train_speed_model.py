
# Train a simple RandomForestRegressor to predict segment speeds (kph)
# Uses synthetic samples for demonstration. Replace with your real measurements for better accuracy.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

np.random.seed(42)

# Load synthetic dataset
try:
    df = pd.read_csv('sample_data/synthetic_speed_samples.csv')
except Exception:
    # generate synthetic samples if CSV missing
    n = 2000
    highway_types = ['motorway','trunk','primary','secondary','tertiary','residential','service']
    hw = np.random.choice(highway_types, size=n, p=[0.2,0.2,0.2,0.15,0.1,0.1,0.05])
    length_m = np.random.gamma(shape=2.0, scale=500.0, size=n) + 50
    is_bridge = np.random.binomial(1, 0.1, size=n)
    base_speed = np.array([
        {'motorway':90,'trunk':80,'primary':60,'secondary':50,'tertiary':40,'residential':30,'service':25}[t]
        for t in hw
    ])
    # synthetic true speed with noise & penalties
    true_speed = base_speed - 5*is_bridge - 0.005*(np.clip(length_m,0,10000)) + np.random.normal(0,4,size=n)
    df = pd.DataFrame({'highway':hw,'length_m':length_m,'is_bridge':is_bridge,'speed_kph':true_speed})
    df.to_csv('sample_data/synthetic_speed_samples.csv', index=False)

# Encode highway to ordinal
code_map = {'motorway':6,'trunk':5,'primary':4,'secondary':3,'tertiary':2,'residential':1,'service':0}
X = df[['length_m','is_bridge']].copy()
X['highway_code'] = df['highway'].map(code_map).fillna(1)
Y = df['speed_kph']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
print(f"R2 on synthetic test set: {r2:.3f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/speed_model.pkl')
print("Saved models/speed_model.pkl")
