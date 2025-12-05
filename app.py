import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load Model Paket Komplit
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        return None

data = load_model()

if data is None:
    st.error("File 'diabetes_model.pkl' tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

model = data["model"]
scaler = data["scaler"]
selected_features = data["selected_features"]
all_columns = data["all_columns"]

st.title("Diabetes Risk Prediction (XGBoost)")
st.markdown("Fill the form to see the result analytics")
st.markdown("---")

# --- DEFINISI DICTIONARY (Mapping Angka ke Teks) ---

# A. Mapping Umur
age_dict = {
    1: "18 - 24 years",
    2: "25 - 29 years",
    3: "30 - 34 years",
    4: "35 - 39 years",
    5: "40 - 44 years",
    6: "45 - 49 years",
    7: "50 - 54 years",
    8: "55 - 59 years",
    9: "60 - 64 years",
    10: "65 - 69 years",
    11: "70 - 74 years",
    12: "75 - 79 years",
    13: "80 years or older"
}

# B. Mapping Pendidikan
edu_dict = {
    1: "Never attended school / Kindergarten only",
    2: "Elementary (Grades 1-8)",
    3: "Some High School (Grades 9-11)",
    4: "High School Graduate / GED",
    5: "Some College / Technical School (1-3 years)",
    6: "College Graduate (4 years or more)"
}

# C. Mapping Pendapatan (BALIK KE USD)
income_dict = {
    1: "Less than $10,000",
    2: "$10,000 - $15,000",
    3: "$15,000 - $20,000",
    4: "$20,000 - $25,000",
    5: "$25,000 - $35,000",
    6: "$35,000 - $50,000",
    7: "$50,000 - $75,000",
    8: "$75,000 or more"
}

# 2. Form Input User
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Medical History")
    high_bp = st.selectbox("High Blood Pressure?", ["No", "Yes"])
    high_chol = st.selectbox("High Cholesterol?", ["No", "Yes"])
    chol_check = st.selectbox("Cholesterol Check in 5 Years?", ["No", "Yes"])
    stroke = st.selectbox("Ever had a Stroke?", ["No", "Yes"])
    heart_dis = st.selectbox("Heart Disease / Attack?", ["No", "Yes"])

with col2:
    st.subheader("Lifestyle")
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 100.0, 25.0)
    alcohol = st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"], help="Men > 14 drinks/week, Women > 7 drinks/week")
    diff_walk = st.selectbox("Difficulty Walking/Climbing Stairs?", ["No", "Yes"])
    gen_hlth = st.selectbox("General Health", 
                            options=[1, 2, 3, 4, 5], 
                            format_func=lambda x: {1:"Excellent", 2:"Very Good", 3:"Good", 4:"Fair", 5:"Poor"}[x])

with col3:
    st.subheader("Demographics")
    sex = st.selectbox("Sex", ["Female", "Male"])
    
    # Selectbox untuk Age, Edu, Income
    age = st.selectbox("Age Group", options=list(age_dict.keys()), format_func=lambda x: age_dict[x])
    edu = st.selectbox("Education Level", options=list(edu_dict.keys()), format_func=lambda x: edu_dict[x])
    income = st.selectbox("Annual Household Income", options=list(income_dict.keys()), format_func=lambda x: income_dict[x])

# Fungsi Konversi
def to_binary(val):
    return 1 if val == "Yes" else 0

sex_val = 1 if sex == "Male" else 0

if st.button("🔍 Predict Risk", type="primary"):
    
    # 3. Persiapan Data
    input_data = {col: 0 for col in all_columns}
    
    input_data['HighBP'] = to_binary(high_bp)
    input_data['HighChol'] = to_binary(high_chol)
    input_data['CholCheck'] = to_binary(chol_check)
    input_data['BMI'] = bmi
    input_data['Stroke'] = to_binary(stroke)
    input_data['HeartDiseaseorAttack'] = to_binary(heart_dis)
    input_data['HvyAlcoholConsump'] = to_binary(alcohol)
    input_data['GenHlth'] = gen_hlth
    input_data['DiffWalk'] = to_binary(diff_walk)
    input_data['Sex'] = sex_val
    input_data['Age'] = age
    input_data['Education'] = edu
    input_data['Income'] = income
    
    df_input = pd.DataFrame([input_data])
    
    # 4. Scaling & Prediction
    df_input = df_input[all_columns] 
    input_scaled = scaler.transform(df_input)
    df_scaled = pd.DataFrame(input_scaled, columns=all_columns)
    
    X_final = df_scaled[selected_features]
    
    prediction = model.predict(X_final)[0]
    probability = model.predict_proba(X_final)[0][1]
    
    st.markdown("---")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("### RESULT: HIGH RISK")
            st.metric("Diabetes Probability", f"{probability*100:.1f}%")
        else:
            st.success("### RESULT: HEALTHY")
            st.metric("Diabetes Probability", f"{probability*100:.1f}%")
            
    with res_col2:
        if prediction == 1:
            st.warning("**AI Analysis:** Based on your inputs, your profile shares patterns with diabetic patients. Please consult a doctor for a blood sugar test.")
        else:
            st.info("**AI Analysis:** Your health profile indicates a low risk. Keep up the healthy lifestyle!")