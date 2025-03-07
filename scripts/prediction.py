import joblib
import numpy as np

def predict_aqi(input_data):
    # Load the trained model.
    model = joblib.load("D:/Aqi_hacathon/AQI_Project/models/aqi_model.pkl")
    
    # Convert the input data to a NumPy array (ensure input_data is a list or array of features).
    features = np.array([input_data])
    
    # Predict the AQI.
    predicted_aqi = model.predict(features)[0]
    print(f"Predicted AQI: {predicted_aqi:.2f}")
    return predicted_aqi

if __name__ == '__main__':
    # Example input: [pm25, pm10, no2, so2, o3, co, latitude, longitude]
    sample_input = [65.2, 98.7, 45.8, 12.3, 38.7, 1.2, 12.9166, 77.6101]
    predict_aqi(sample_input)
