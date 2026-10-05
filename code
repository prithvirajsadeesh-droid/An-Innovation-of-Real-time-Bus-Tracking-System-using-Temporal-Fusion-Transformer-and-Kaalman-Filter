import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

from kalman_filter import KalmanFilter1D
from database import create_database, insert_bus_data, get_latest_data


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Real-Time Bus Tracking System",
    page_icon="🚌",
    layout="wide"
)


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

create_database()

MODEL_PATH = "models/random_forest_eta.pkl"

model = joblib.load(MODEL_PATH)

latitude_filter = KalmanFilter1D(
    process_variance=0.00001,
    measurement_variance=0.001
)

longitude_filter = KalmanFilter1D(
    process_variance=0.00001,
    measurement_variance=0.001
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🚌 Real-Time Bus Tracking System")

st.subheader(
    "Using Kalman Filter and Random Forest"
)

st.write(
    "The system filters noisy GPS data using a Kalman Filter "
    "and predicts Estimated Time of Arrival (ETA) using "
    "Random Forest."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Bus Information")

bus_id = st.sidebar.text_input(
    "Bus ID",
    "BUS-101"
)

latitude = st.sidebar.number_input(
    "GPS Latitude",
    value=10.7905,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "GPS Longitude",
    value=78.7047,
    format="%.6f"
)

speed = st.sidebar.slider(
    "Bus Speed (km/h)",
    10,
    80,
    40
)

traffic = st.sidebar.slider(
    "Traffic Level",
    1,
    5,
    3
)

distance = st.sidebar.slider(
    "Distance to Destination (km)",
    1.0,
    30.0,
    10.0
)


# --------------------------------------------------
# GPS FILTERING
# --------------------------------------------------

filtered_latitude = latitude_filter.update(latitude)

filtered_longitude = longitude_filter.update(longitude)


# --------------------------------------------------
# HISTORICAL TRAVEL TIME
# --------------------------------------------------

if speed > 0:

    historical_time = (
        distance / speed
    ) * 60

else:

    historical_time = 0


# --------------------------------------------------
# RANDOM FOREST ETA PREDICTION
# --------------------------------------------------

input_data = pd.DataFrame({
    "distance": [distance],
    "speed": [speed],
    "traffic": [traffic],
    "historical_time": [historical_time]
})


eta = model.predict(input_data)[0]

eta = max(0, eta)


# --------------------------------------------------
# DATABASE STORAGE
# --------------------------------------------------

if st.sidebar.button("Update Bus Location"):

    insert_bus_data(
        bus_id,
        filtered_latitude,
        filtered_longitude,
        speed,
        traffic,
        distance,
        eta
    )

    st.success(
        "Bus location and ETA updated successfully!"
    )


# --------------------------------------------------
# CURRENT BUS STATUS
# --------------------------------------------------

st.header("Current Bus Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Bus ID",
        bus_id
    )

with col2:
    st.metric(
        "Speed",
        f"{speed:.1f} km/h"
    )

with col3:
    st.metric(
        "Distance",
        f"{distance:.1f} km"
    )

with col4:
    st.metric(
        "Predicted ETA",
        f"{eta:.1f} min"
    )


# --------------------------------------------------
# GPS LOCATION
# --------------------------------------------------

st.header("📍 Filtered GPS Location")

location_data = pd.DataFrame({
    "Latitude": [filtered_latitude],
    "Longitude": [filtered_longitude]
})

st.map(
    location_data,
    latitude="Latitude",
    longitude="Longitude"
)


# --------------------------------------------------
# SYSTEM INFORMATION
# --------------------------------------------------

st.header("⚙️ Prediction Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write("### Input Features")

    st.write(
        f"Distance: {distance:.2f} km"
    )

    st.write(
        f"Speed: {speed:.2f} km/h"
    )

    st.write(
        f"Traffic Level: {traffic}"
    )

    st.write(
        f"Historical Travel Time: "
        f"{historical_time:.2f} minutes"
    )


with info_col2:

    st.write("### Algorithms Used")

    st.write(
        "✅ Kalman Filter"
    )

    st.write(
        "✅ Random Forest"
    )

    st.write(
        "✅ Real-Time GPS Processing"
    )

    st.write(
        "✅ SQLite Database"
    )


# --------------------------------------------------
# DATABASE HISTORY
# --------------------------------------------------

st.header("📊 Recent Bus Tracking Data")

latest_data = get_latest_data()

if latest_data:

    columns = [
        "ID",
        "Bus ID",
        "Latitude",
        "Longitude",
        "Speed",
        "Traffic",
        "Distance",
        "ETA",
        "Timestamp"
    ]

    history_df = pd.DataFrame(
        latest_data,
        columns=columns
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:

    st.info(
        "No tracking data available yet."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Real-Time Bus Tracking System | "
    "Random Forest + Kalman Filter"
)
