
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction Using Machine Learning")
st.write(
    "Analyze crop data and estimate crop yield using rainfall, "
    "temperature, soil fertility and fertilizer usage."
)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

DATA_FILE = "/content/crop_yield.csv"
HISTORY_FILE = "/content/prediction_history.csv"

if not os.path.exists(DATA_FILE):
    st.error("Dataset not found: /content/crop_yield.csv")
    st.info("Upload your crop_yield.csv file to Google Colab.")
    st.stop()

df = pd.read_csv(DATA_FILE)

features = [
    "Rainfall_mm",
    "Temperature_C",
    "Soil_Fertility",
    "Fertilizer_kg"
]

target = "Crop_Yield_ton"

required_columns = features + [target]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    st.stop()

# --------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------

df = df[required_columns].copy()

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

original_rows = len(df)

df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

st.sidebar.header("🔍 Data Filters")
st.sidebar.caption(f"Rows available: {len(df)}")
st.sidebar.caption(f"Rows removed during cleaning: {original_rows - len(df)}")

if df.empty:
    st.error("No valid records are available after data cleaning.")
    st.stop()

# --------------------------------------------------
# 3. DATA FILTERING
# --------------------------------------------------

selected_column = st.sidebar.selectbox(
    "Select a column",
    required_columns
)

minimum = float(df[target].min())
maximum = float(df[target].max())

if minimum < maximum:
    yield_range = st.sidebar.slider(
        "Select crop yield range (tons)",
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum)
    )

    filtered_df = df[
        df[target].between(yield_range[0], yield_range[1])
    ]
else:
    filtered_df = df.copy()
    st.sidebar.info("Only one crop yield value is available.")

st.subheader("📊 Project Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(filtered_df))
col2.metric(
    "Average Yield",
    f"{filtered_df[target].mean():.2f} tons"
)
col3.metric(
    "Maximum Yield",
    f"{filtered_df[target].max():.2f} tons"
)

# --------------------------------------------------
# 4. DATASET TABLE
# --------------------------------------------------

st.subheader("📋 Crop Dataset")
st.dataframe(filtered_df, use_container_width=True)

# --------------------------------------------------
# 5. SUMMARY STATISTICS
# --------------------------------------------------

st.subheader("📊 Summary Statistics")
st.dataframe(
    filtered_df.describe().round(3),
    use_container_width=True
)

# --------------------------------------------------
# 6. DATA VISUALIZATION
# --------------------------------------------------

st.subheader("📈 Data Visualization")

st.write(f"Selected column: {selected_column}")

fig, ax = plt.subplots()

ax.hist(
    filtered_df[selected_column],
    bins=8,
    edgecolor="black"
)

ax.set_title(f"Distribution of {selected_column}")
ax.set_xlabel(selected_column)
ax.set_ylabel("Frequency")

st.pyplot(fig)
plt.close(fig)

# Rainfall vs Crop Yield
st.subheader("🌧️ Rainfall vs Crop Yield")

fig, ax = plt.subplots()

ax.scatter(
    df["Rainfall_mm"],
    df[target]
)

ax.set_title("Rainfall and Crop Yield Relationship")
ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Crop Yield (tons)")

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 7. TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

st.subheader("🤖 Predict Crop Yield")

X = df[features]
y = df[target]

if len(df) < 5:
    st.warning(
        "At least 5 valid records are recommended for model training."
    )
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predicted_test = model.predict(X_test)

metric1, metric2 = st.columns(2)

metric1.metric(
    "Model R² Score",
    f"{r2_score(y_test, predicted_test):.3f}"
)

metric2.metric(
    "Mean Absolute Error",
    f"{mean_absolute_error(y_test, predicted_test):.3f} tons"
)

st.caption(
    "Model evaluation uses a held-out test set. "
    "A small dataset may produce unreliable scores."
)

# --------------------------------------------------
# 8. USER INPUTS
# --------------------------------------------------

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=5000.0,
    value=float(df["Rainfall_mm"].median())
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=float(df["Temperature_C"].median())
)

soil = st.slider(
    "Soil Fertility",
    min_value=0,
    max_value=100,
    value=int(df["Soil_Fertility"].median())
)

fertilizer = st.number_input(
    "Fertilizer (kg)",
    min_value=0.0,
    max_value=1000.0,
    value=float(df["Fertilizer_kg"].median())
)

# --------------------------------------------------
# 9. PREDICTION AND CSV STORAGE
# --------------------------------------------------

if st.button("Predict Crop Yield"):

    input_data = pd.DataFrame(
        [[rainfall, temperature, soil, fertilizer]],
        columns=features
    )

    prediction = float(model.predict(input_data)[0])

    st.success(
        f"Estimated Crop Yield: {prediction:.2f} tons"
    )

    record = pd.DataFrame([{
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "Soil_Fertility": soil,
        "Fertilizer_kg": fertilizer,
        "Predicted_Yield_ton": round(prediction, 2)
    }])

    if os.path.exists(HISTORY_FILE):
        record.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        record.to_csv(
            HISTORY_FILE,
            index=False
        )

    st.success("Prediction saved successfully!")

# --------------------------------------------------
# 10. PREDICTION HISTORY
# --------------------------------------------------

st.subheader("📁 Prediction History")

if os.path.exists(HISTORY_FILE):

    history = pd.read_csv(HISTORY_FILE)

    st.dataframe(
        history,
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Prediction History",
        data=history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

else:
    st.info(
        "No predictions saved yet. "
        "Click Predict Crop Yield to create your first record."
    )

st.divider()

st.caption(
    "Educational demonstration only. "
    "Predictions from a small or synthetic dataset should not "
    "be used as actual agricultural recommendations."
)
