# app_data_processing.py
import streamlit as st
import pandas as pd

# 0) Streamlit page config
st.set_page_config(page_title="🌱 AI-Powered Crop Advisor", layout="centered")

# 1) Load data
@st.cache_data
def load_data(path="sustainable_crop_recommendation.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # remove extra spaces in column names
    return df

df = load_data()

# 2) Preview data
st.title("🌱 Sustainable Agriculture Advisor (Data Preview)")
st.subheader("👀 First 10 rows of dataset")
st.dataframe(df.head(10))

# 3) Identify feature columns
def find_col(candidates):
    """Find first matching column from a list of candidates"""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

col_N = find_col(["N"])
col_P = find_col(["P"])
col_K = find_col(["K"])
col_temp = find_col(["temperature", "temp"])
col_hum = find_col(["humidity"])
col_ph = find_col(["pH", "ph"])
col_rain = find_col(["rainfall"])

feature_cols = [c for c in [col_N, col_P, col_K, col_temp, col_hum, col_ph, col_rain] if c]

# 4) Identify target column
main_crop = find_col(["label", "crop", "crops"])

st.subheader("🔎 Detected Columns")
st.write("Feature columns:", feature_cols)
st.write("Target column:", main_crop)

# 5) Convert features to numeric
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = df[main_crop].astype(str)

# 6) Remove rows with missing values
valid = X.notna().all(axis=1)
X = X[valid]
y = y[valid]

st.subheader("✅ Data after cleaning")
st.write(f"Total valid rows: {len(X)}")
st.dataframe(pd.concat([X, y], axis=1).head(10))
