import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_analysis():
    print("Performing Exploratory Data Analysis on cleaned data...")
    df = pd.read_csv("D:/Aqi_hacathon/AQI_Project/data/cleaned_aqi_data.csv")
    
    # Display the first few rows and statistics.
    print("Data Head:")
    print(df.head())
    print("\nData Description:")
    print(df.describe())
    
    # Plot a correlation heatmap for numeric columns.
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

if __name__ == '__main__':
    exploratory_analysis()
