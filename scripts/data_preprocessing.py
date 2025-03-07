import pandas as pd

def preprocess_data():
    print("\n🚀 Step 1: Data Preprocessing Started...\n")
    
    # Load the raw sample data.
    df = pd.read_csv("D:/Aqi_hacathon/AQI_Project/data/sample_aqi_data.csv")
    
    # Convert the 'date' column to datetime.
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Fill missing numeric values with the column mean.
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Drop duplicate rows.
    df.drop_duplicates(inplace=True)
    
    # Save the cleaned data.
    df.to_csv("D:/Aqi_hacathon/AQI_Project/data/cleaned_aqi_data.csv", index=False)
    print("✅ Data Preprocessed & Saved as 'data/cleaned_aqi_data.csv'")
    return df

if __name__ == '__main__':
    preprocess_data()
