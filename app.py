from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import joblib
import numpy as np
import math
import winsound

app = Flask(__name__)

# Load the AI model once when the app starts
model = joblib.load("D:/Aqi_hacathon/AQI_Project/models/aqi_model.pkl")

# Dummy historical AQI data for API endpoints
historical_aqi_data = [
    {
        "id": 1,
        "date": "2025-02-01",
        "aqi": 156,
        "pm25": 42,
        "pm10": 75,
        "o3": 0.045,
        "no2": 0.052,
        "so2": 0.012,
        "co": 1.2
    },
    {
        "id": 2,
        "date": "2025-02-15",
        "aqi": 162,
        "pm25": 45,
        "pm10": 80,
        "o3": 0.048,
        "no2": 0.055,
        "so2": 0.015,
        "co": 1.3
    },
    {
        "id": 3,
        "date": "2025-03-01",
        "aqi": 145,
        "pm25": 38,
        "pm10": 68,
        "o3": 0.042,
        "no2": 0.048,
        "so2": 0.010,
        "co": 1.1
    }
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/sample_data', methods=['POST'])
def sample_data():
    try:
        location = request.form.get('location')
        date_str = request.form.get('date')

        if not location or not date_str:
            return jsonify({"error": "Missing required fields: location and date"}), 400

        # Load the dataset
        df = pd.read_csv("D:/Aqi_hacathon/AQI_Project/data/sample_aqi_data.csv")

        # Filter data for location & date
        filtered = df[(df['station_name'].str.lower() == location.lower()) & (df['date'] == date_str)]

        if not filtered.empty:
            # Convert to HTML table
            table_html = filtered.to_html(classes="aqi-table", index=False)
            data_for_js = filtered.to_dict(orient='records')
            # Use the AQI value from the CSV for this city and date
            aqi_value = filtered['aqi'].iloc[0]
            return render_template('sample_data.html',
                                   location=location,
                                   date=date_str,
                                   tables=[table_html],
                                   aqi_data=json.dumps(data_for_js),
                                   predicted_aqi=None,
                                   aqi_value=aqi_value,
                                   message=None)

        # If no real data, predict AQI using the AI model
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')

        if lat is None or lon is None:
            return jsonify({"error": "Missing required fields: latitude and longitude"}), 400

        try:
            lat = float(lat)
            lon = float(lon)
        except ValueError:
            return jsonify({"error": "Invalid latitude or longitude values"}), 400

        # Use default pollutant values if not provided (modify as needed)
        default_features = [50.0, 80.0, 30.0, 10.0, 40.0, 1.0, lat, lon]
        predicted_aqi = model.predict([default_features])[0]
        aqi_value = round(predicted_aqi, 2)

        prediction_result = {
            "station_name": location,
            "date": date_str,
            "latitude": lat,
            "longitude": lon,
            "aqi": aqi_value,
            "category": "Predicted AQI",
            "dominant_pollutant": "N/A"
        }

        return render_template('sample_data.html',
                               location=location,
                               date=date_str,
                               tables=[],  # No table data available
                               aqi_data=json.dumps([prediction_result]),
                               predicted_aqi=aqi_value,
                               aqi_value=aqi_value,
                               message=f"Predicted AQI: {predicted_aqi:.2f}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Added Backend Code for Third Frontend Page ---------------------
@app.route('/aqi_solutions.html')
def pollution_sources():
    # Retrieve query parameters from the URL
    location = request.args.get('location')
    date_str = request.args.get('date')
    
    # Load the sample_aqi_data.csv as your database
    try:
        df = pd.read_csv("D:/Aqi_hacathon/AQI_Project/data/sample_aqi_data.csv")
    except Exception as e:
        return jsonify({"error": f"Error loading CSV: {str(e)}"}), 500

    # Filter the dataset for the selected location and date
    if location and date_str:
        filtered = df[(df['station_name'].str.lower() == location.lower()) & (df['date'] == date_str)]
    else:
        filtered = pd.DataFrame()

    # Use the AQI from the CSV if available; otherwise, default to 100
    if not filtered.empty:
        try:
            aqi_value = float(filtered['aqi'].iloc[0])
        except Exception:
            aqi_value = 100
    else:
        aqi_value = 100  # Default value if no matching data is found

    # Calculate estimated trees needed based on the retrieved AQI
    trees_needed = calculate_trees_needed(aqi_value)
    
    # Determine likely pollution sources based on the retrieved AQI
    pollution_sources_data = determine_pollution_sources(aqi_value)
    
    return render_template('pollution_sources.html',
                           location=location,
                           date=date_str,
                           aqi=aqi_value,
                           trees_needed=trees_needed,
                           pollution_sources=pollution_sources_data)
# -----------------------------------------------------------------------------------------

# --------------------- Helper Functions for Third Page ---------------------
def calculate_trees_needed(aqi):
    """
    Calculate estimated number of trees needed to counter pollution based on AQI.
    This is a simplified model.
    """
    if aqi <= 50:
        return int(aqi * 0.5)
    elif aqi <= 100:
        return int(aqi * 1)
    elif aqi <= 150:
        winsound.Beep(1000, 1000)
        return int(aqi * 2)
    elif aqi <= 200:
        winsound.Beep(1000, 1000)
        return int(aqi * 3)
    elif aqi <= 300:
        winsound.Beep(1000, 1000)
        return int(aqi * 5)
    else:
        winsound.Beep(1000, 1000)
        return int(aqi * 8)



def logistic(x):
    return 1 / (1 + math.exp(-x))

def determine_pollution_sources(aqi):
    """
    Determine likely pollution sources based on AQI using a more complex model.
    This model uses logistic functions to simulate nonlinear contributions of each source.
    """
    # Raw contributions calculated with logistic functions:
    natural = 1 - logistic(0.1 * (aqi - 50))        # Decreases as AQI increases
    traffic = logistic(0.1 * (aqi - 40))              # Starts increasing at moderate AQI
    industrial = logistic(0.15 * (aqi - 90))          # Increases sharply after ~90
    residential = logistic(0.1 * (100 - abs(aqi - 100)))  # Peaks around 100
    construction = logistic(0.11 * (aqi - 100)) if aqi > 100 else 0
    power_plants = logistic(0.1 * (aqi - 150)) if aqi > 150 else 0
    ag_burning = logistic(0.1 * (aqi - 150)) if aqi > 150 else 0
    wildfires = logistic(0.1 * (aqi - 300)) if aqi > 300 else 0

    # Scale factors (tunable constants) for each source:
    weights = {
        "Natural Sources": natural * 40,
        "Traffic": traffic * 50,
        "Industrial": industrial * 60,
        "Residential": residential * 30,
        "Construction": construction * 20,
        "Power Plants": power_plants * 40,
        "Agricultural Burning": ag_burning * 30,
        "Wildfires/Burning": wildfires * 30,
    }
    
    # Remove negligible contributions:
    weights = {k: v for k, v in weights.items() if v > 0.1}
    
    total = sum(weights.values())
    if total == 0:
        # Fallback distribution if something goes wrong.
        return {"Traffic": 40, "Industrial": 30, "Residential": 20, "Natural Sources": 10}
    
    # Normalize to sum 100:
    percentages = {k: round((v / total) * 100) for k, v in weights.items()}
    return percentages

# -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
