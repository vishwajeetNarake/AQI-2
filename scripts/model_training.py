import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model():
    print("\n🤖 Step 3: Training AI Model...\n")
    
    # Load cleaned data.
    df = pd.read_csv("D:/Aqi_hacathon/AQI_Project/data/cleaned_aqi_data.csv")
    
    # Select features and target.
    # Ensure your CSV columns are in lowercase (or adjust accordingly).
    features = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co', 'latitude', 'longitude']
    target = 'aqi'
    X = df[features]
    y = df[target]
    
    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Linear Regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model.
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")
    
    # Save the trained model.
    joblib.dump(model, "D:/Aqi_hacathon/AQI_Project/models/aqi_model.pkl")
    print("✅ Model Trained & Saved as 'models/aqi_model.pkl'")
    
if __name__ == '__main__':
    train_model()
