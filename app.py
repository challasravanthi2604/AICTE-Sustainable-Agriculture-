# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# 0) MUST be the first Streamlit command
st.set_page_config(page_title="🌱 AI-Powered Crop Advisor", layout="centered")

# 1) Load data
@st.cache_data
def load_data(path="sustainable_crop_recommendation.csv"):
    return pd.read_csv(path)

df = load_data()
df.columns = df.columns.str.strip()

#st.subheader("🔎 Detected columns in your CSV")
#st.write(df.columns.tolist())

# 2) Helpers to find columns regardless of exact spelling/case
def find_col(candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

# Detect feature columns
col_N = find_col(["N"])
col_P = find_col(["P"])
col_K = find_col(["K"])
col_temp = find_col(["temperature", "temp"])
col_hum = find_col(["humidity"])
col_ph = find_col(["pH", "ph"])
col_rain = find_col(["rainfall"])

feature_cols = [c for c in [col_N, col_P, col_K, col_temp, col_hum, col_ph, col_rain] if c]

# Detect target/output columns
main_crop = find_col(["label", "crop", "crops"])
mixed_crop = find_col(["mixed_crops", "Mixed_Crops"])
fert_col = find_col(["organic_fertilizer", "Organic_Fertilizer"])
bio_col = find_col(["biopesticide", "Biopesticide"])
tree_col = find_col(["agroforestry_tree", "Agroforestry_Tree"])

target_cols = [c for c in [main_crop, mixed_crop, fert_col, bio_col, tree_col] if c]

if not target_cols:
    st.error("❌ No target/output columns found. Make sure CSV has Crop, Mixed_Crops, Organic_Fertilizer, Biopesticide, Agroforestry_Tree.")
    st.stop()

# Validate features
missing = []
if not col_N: missing.append("N")
if not col_P: missing.append("P")
if not col_K: missing.append("K")
if not col_temp: missing.append("temperature")
if not col_hum: missing.append("humidity")
if not col_ph: missing.append("pH")
if not col_rain: missing.append("rainfall")

if missing:
    st.error(f"❌ Missing expected feature columns: {missing}")
    st.stop()

# 3) Build X, y
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = df[target_cols].astype(str)

valid = X.notna().all(axis=1)
X = X[valid]
y = y[valid]

if len(X) < 50:
    st.error("❌ Too few valid rows after cleaning. Check your CSV values/types.")
    st.stop()

# 4) Train multi-output model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y[main_crop] if main_crop else None
)

base_model = RandomForestClassifier(n_estimators=200, random_state=42)
model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

# Accuracy for main crop only (since multi-output accuracy is tricky)
main_acc = accuracy_score(y_test[main_crop], model.predict(X_test)[:,0]) * 100

st.title("🌱 Sustainable Agriculture Advisor (AI-Powered)")
st.caption("Predicts the best crop + companion strategies for your soil & climate.")
#st.success(f"✅ Main Crop Accuracy: **{main_acc:.2f}%**")

# 5) Sidebar inputs
st.sidebar.header("🧪 Enter Soil & Climate")
N = st.sidebar.slider("Nitrogen (N)", 0, 150, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 150, 50)
K = st.sidebar.slider("Potassium (K)", 0, 200, 50)
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 27)
humidity = st.sidebar.slider("Humidity (%)", 20, 100, 70)
ph = st.sidebar.slider("Soil pH", 4.0, 9.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 100, 3000, 1200)

if st.button("🌾 Recommend Package"):
    input_row = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]], 
        columns=feature_cols
    )
    input_row = input_row.reindex(columns=X.columns, fill_value=0)

    preds = model.predict(input_row)[0]

    #st.balloons()
    st.success(f"🌾 Recommended Crop: **{preds[0]}**")

    if len(preds) > 1:
        st.info(f"🤝 Suggested Mixed Crop: **{preds[1]}**")
    if len(preds) > 2:
        st.info(f"🌿 Organic Fertilizer: **{preds[2]}**")
    if len(preds) > 3:
        st.info(f"🪲 Biopesticide: **{preds[3]}**")
    if len(preds) > 4:
        st.info(f"🌳 Agroforestry Tree: **{preds[4]}**")

with st.expander("👀 Preview data"):
    st.dataframe(df.head(10))
