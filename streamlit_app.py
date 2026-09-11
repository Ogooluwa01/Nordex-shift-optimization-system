import streamlit as st
import pandas as pd
import requests
import datetime

from src.data.data_ingestion import load_data

API_URL = "https://localhost:8000"

st.set_page_config(
    page_title= "Nordex Shift Optimization",
    layout = "wide"
)

# Load Categorical values
@st.cache_data
def get_categorical_values():
    df=load_data
    return {
        "shift_name": sorted(df["shift_name"].dropna().unique().tolist()),
        "skill_category": sorted(df["skill_category"].dropna().unique().tolist()),
        "machine_status": sorted(df["machine_status"].dropna().unique().tolist())
    }

categories = get_categorical_values()

# header 
st.title("Nordex Shift Performance Dashboard")
st.markdown("""
Predict manufacturinmg shift performance, explore optimal operating conditions, and manage model retraining. """)

## Tabs
prediction_tab, optimization_tab, retraining_tab = st.tabs([
    "Shift Prediction",
    "Shift Optimization",
    "Model Retraining "
])
# shift prediction
with prediction_tab:
    st.header("Shift Performance Prediction")
    st.write("Enter current shift information to predict expected shift efficiency")
    st.divider()

    st.subheader("Production Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        units_produced= st.number_input("Units Produced", min_value=0, max_value=2000,value=800, step=1 )

    with col2:
        defect_count= st.number_input("Defect Count", min_value = 0, max_value=500, value= 20, step=1)

    with col3:
        cycle_time_avg = st.number_input("Average Cycle Time", min_value = 0, max_value=100.0, value=15.0, step=0.5 )


    st.subheader("Workforce & Runtime")
    col1, col2, col3 = st.columns(3)

    with col1:
        experience_level = st.slider("Experience Level", min_value=1, max_value=15,value=6 )

    with col2:
        runtime_hours = st.number_input("Runtime Hours", min_value=0.0, max_value=24.0,value=7.5, step=0.5 )

    with col3:
        downtime_minutes = st.number_input("Downtime Minutes", min_value=0, max_value=300.0,value=30.0, step=1.0)

    st.subheader("Maintenance")
    col1, col2 = st.columns(2)

    with col1:
        maintenance_flag = st.selectbox("Maintenance Flag", [0,1], format_func=lambda x: "Yes" if x ==1 else "No" )

    with col2:
        maintenance_downtime = st.number_input("Mainteanace Downtime", min_value=0.0, max_value=180.0, value=0.0, step=1)

    st.subheader("Environmental Conditions")
    col1, col2 = st.colums(2)

    with col1:
        temperature = st.number_input("Temperature", min_value =15.0, max_value= 40.0, value=22.0, step =0.5 )

    with col2:
        humidity = st.number_input("Humidity", min_value=10.0, max_value = 100.0, value =50.0, step=1.0)

        

