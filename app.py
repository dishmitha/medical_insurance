"""
Medical Insurance Cost Prediction App
=====================================
Streamlit application following ML pipeline steps:
1. Load Dataset
2. Understand Dataset
3. Find Input and Output & Visualize
4. Divide Data (Train/Test Split)
5. Create Model & Train
6. Testing
7. Prediction
8. UI using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None

# Title
st.title("🏥 Medical Insurance Cost Prediction")
st.markdown("---")

# ============================================
# STEP 1: LOAD DATASET
# ============================================
st.header("📁 Step 1: Load Dataset")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success("✅ Dataset loaded successfully!")
else:
    # Check if default dataset exists
    if os.path.exists('medical_insurance_data.csv'):
        df = pd.read_csv('medical_insurance_data.csv')
        st.session_state.df = df
        st.info("📂 Using default dataset: medical_insurance_data.csv")
    else:
        st.warning("⚠️ Please upload a CSV file to proceed!")
        st.stop()

# ============================================
# STEP 2: UNDERSTAND DATASET
# ============================================
st.markdown("---")
st.header("🔍 Step 2: Understand Dataset")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Dataset Shape")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")

with col2:
    st.subheader("Column Names")
    st.write(df.columns.tolist())

st.subheader("Data Types")
st.dataframe(df.dtypes.astype(str), width='stretch')

st.subheader("First 5 Rows")
st.dataframe(df.head())

st.subheader("Statistical Summary")
st.dataframe(df.describe())

st.subheader("Missing Values")
missing = df.isnull().sum()
if missing.sum() == 0:
    st.success("✅ No missing values!")
else:
    st.write(missing[missing > 0])

# ============================================
# STEP 3: FIND INPUT & OUTPUT & VISUALIZE
# ============================================
st.markdown("---")
st.header("📊 Step 3: Find Input & Output & Visualize Data")

# Identify target column
target_col = st.selectbox("Select Target Variable (Output)", df.columns, index=len(df.columns)-1)

# Identify input columns
feature_cols = [col for col in df.columns if col != target_col]
st.subheader("Input Features (X)")
st.write(feature_cols)

# Visualizations
st.subheader("Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Target distribution
    fig, ax = plt.subplots()
    ax.hist(df[target_col], bins=30, color='steelblue', edgecolor='black')
    ax.set_xlabel(target_col)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{target_col} Distribution')
    st.pyplot(fig)

with col2:
    # Correlation with numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        correlations = df[numeric_cols].corr()[target_col].drop(target_col).sort_values(ascending=False)
        fig, ax = plt.subplots()
        correlations.plot(kind='bar', ax=ax, color='coral')
        ax.set_title(f'Correlation with {target_col}')
        ax.set_ylabel('Correlation')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Show categorical distributions
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    st.subheader("Categorical Features Distribution")
    col1, col2 = st.columns(2)
    for i, col in enumerate(cat_cols):
        with col1 if i % 2 == 0 else col2:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'coral', 'green', 'orange', 'purple'])
            ax.set_title(f'{col} Distribution')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)

# ============================================
# STEP 4: DIVIDE DATA (TRAIN/TEST SPLIT)
# ============================================
st.markdown("---")
st.header("📂 Step 4: Train/Test Split")

test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
random_state = st.number_input("Random State", 0, 100, 42)

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

st.write(f"**Training samples:** {len(X_train)}")
st.write(f"**Testing samples:** {len(X_test)}")

# ============================================
# STEP 5: CREATE MODEL & TRAIN
# ============================================
st.markdown("---")
st.header("🤖 Step 5: Create Model & Train")

# Preprocessing - Encode categorical variables
df_train = X_train.copy()
df_test = X_test.copy()

label_encoders = {}
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le

# Store in session state
st.session_state.label_encoders = label_encoders

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train)
X_test_scaled = scaler.transform(df_test)

st.session_state.scaler = scaler

# Create and train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

st.session_state.model = model
st.session_state.X_columns = feature_cols

st.success("✅ Model trained successfully!")

# ============================================
# STEP 6: TESTING (MODEL EVALUATION)
# ============================================
st.markdown("---")
st.header("✅ Step 6: Testing - Model Evaluation")

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Display metrics
st.subheader("Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("R² Score (Train)", f"{train_r2:.4f}")
with col2:
    st.metric("R² Score (Test)", f"{test_r2:.4f}")
with col3:
    st.metric("RMSE (Train)", f"{train_rmse:.2f}")
with col4:
    st.metric("RMSE (Test)", f"{test_rmse:.2f}")

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.metric("MAE (Train)", f"{train_mae:.2f}")
with col6:
    st.metric("MAE (Test)", f"{test_mae:.2f}")
with col7:
    st.metric("MSE (Train)", f"{train_mse:.2f}")
with col8:
    st.metric("MSE (Test)", f"{test_mse:.2f}")

# Visualization: Actual vs Predicted
st.subheader("Actual vs Predicted (Test Set)")
fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred, alpha=0.5, c='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Actual vs Predicted')
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Coefficients")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
ax.set_xlabel('Coefficient Value')
ax.set_title('Feature Coefficients')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
st.pyplot(fig)

# ============================================
# STEP 7: PREDICTION
# ============================================
st.markdown("---")
st.header("🔮 Step 7: Make Prediction")

st.subheader("Enter Input Values")

# Create input fields for each feature
input_data = {}
col1, col2, col3 = st.columns(3)

for i, col in enumerate(feature_cols):
    with [col1, col2, col3][i % 3]:
        if col in categorical_cols:
            input_data[col] = st.selectbox(f"{col}", df[col].unique())
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(df[col].mean())
            input_data[col] = st.number_input(f"{col}", min_val, max_val, default_val)

# Make prediction
if st.button("Predict"):
    # Prepare input dataframe
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: #1f77b4; margin: 0;">{prediction:,.2f}</h2>
        <p style="color: #666;">Estimated {target_col}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# SIDEBAR - DOWNLOAD & INFO
# ============================================
st.sidebar.markdown("---")
st.sidebar.header("📥 Download")
if os.path.exists('medical_insurance_data.csv'):
    with open("medical_insurance_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="Download Sample CSV",
            data=file,
            file_name="medical_insurance_data.csv",
            mime="text/csv"
        )

st.sidebar.info("""
**Pipeline Steps:**
1. ✅ Load Dataset
2. ✅ Understand Dataset
3. ✅ Find Input/Output & Visualize
4. ✅ Train/Test Split
5. ✅ Create Model & Train
6. ✅ Testing
7. ✅ Prediction
8. ✅ Streamlit UI
""")

