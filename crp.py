import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="🌾 Crop Suitability & Production Dashboard", layout="wide")

# Cache data loading to speed up app
@st.cache_data
def load_data():
    file_path = 'D:/GUVI PROJ03/FAOSTAT_data - FAOSTAT_data_en_12-29-2024.csv'
    df = pd.read_csv(file_path)
    return df

# Data Preprocessing
def preprocess_data(df):
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Area': 'Country', 'Item': 'Crop'})

    # Focus only on relevant elements for crop production
    crop_elements = ['Area harvested', 'Yield', 'Production']

    df = df[df['Element'].isin(crop_elements)]

    # Pivot data so elements become columns
    df_pivot = df.pivot_table(
        index=['Country', 'Crop', 'Year'],
        columns='Element',
        values='Value',
        aggfunc='first'
    ).reset_index()

    df_pivot = df_pivot.fillna(0)  # Replace missing values with zeros (optional)
    return df_pivot

# Crop Suitability Recommendation
def crop_suitability(df, country):
    st.subheader("🌍 Crop Suitability Recommendation")
    country_data = df[df['Country'] == country]

    if country_data.empty:
        st.warning(f"No data available for {country}.")
        return

    # Summarize production data
    crop_totals = country_data.groupby('Crop')['Production'].sum().sort_values(ascending=False)
    st.write(f"Top 5 Crops Based on Total Production in {country}")
    st.bar_chart(crop_totals.head(5))

    # Show detailed crop data
    st.write("Detailed Crop Production Data")
    st.dataframe(crop_totals.reset_index())

# Exploratory Data Analysis with Enhanced Visualizations
def plot_eda(df):
    st.subheader("📊 Exploratory Data Analysis")

    st.write("### Sample Data")
    st.dataframe(df.head())

    # Trend Over Time
    if 'Production' in df.columns:
        st.write("### 📈 Production Trend Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df, x='Year', y='Production', marker='o', ax=ax)
        plt.xticks(rotation=45)
        plt.title('Crop Production Trend Over Time')
        st.pyplot(fig)

    # Outliers Detection - Box Plot
    if 'Production' in df.columns:
        st.write("### 📦 Outlier Detection (Production)")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='Year', y='Production', ax=ax)
        plt.title('Yearly Production - Outlier Detection')
        st.pyplot(fig)

    # Area Harvested vs Production (Spot Patterns)
    if 'Area harvested' in df.columns and 'Production' in df.columns:
        st.write("### 🔎 Area Harvested vs Production (Spot Patterns)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(data=df, x='Area harvested', y='Production', scatter_kws={'s': 60}, line_kws={"color": "red"}, ax=ax)
        plt.title('Area Harvested vs Production (with Regression Line)')
        st.pyplot(fig)

# Model Training Function
def train_model(X_train, X_test, y_train, y_test):
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    return {
        'model': lr,
        'MSE': mean_squared_error(y_test, y_pred_lr),
        'R²': r2_score(y_test, y_pred_lr),
        'MAE': mean_absolute_error(y_test, y_pred_lr)
    }

# Prediction Interface
def prediction_interface(model):
    st.subheader("🔮 Predict Future Crop Production")
    col1, col2, col3 = st.columns(3)

    area = col1.number_input("Area Harvested (ha)", min_value=0.0)
    yield_val = col2.number_input("Yield (kg/ha)", min_value=0.0)
    year = col3.number_input("Year", min_value=2000, max_value=2050)

    if st.button("Predict Production"):
        prediction = model.predict([[area, yield_val, year]])
        st.success(f"🌾 Predicted Production: **{prediction[0]:,.2f} tons**")

# Main App Logic
def main():
    st.title("🌾 Crop Suitability & Production Dashboard")
    st.write("Analyze crop suitability, predict crop production, and explore FAOSTAT data using machine learning models.")

    df = load_data()
    df_clean = preprocess_data(df)

    # Sidebar Filters
    countries = df_clean['Country'].unique()
    country = st.sidebar.selectbox("🌍 Select Country", countries)

    crops = df_clean[df_clean['Country'] == country]['Crop'].unique()
    crop = st.sidebar.selectbox("🌾 Select Crop", crops)

    # Filtered Data
    filtered_df = df_clean[(df_clean['Country'] == country) & (df_clean['Crop'] == crop)]

    st.write(f"**Filtered Data - {len(filtered_df)} records found for {crop} in {country}.**")
    st.dataframe(filtered_df)

    if len(filtered_df) == 0:
        st.error("❌ No data available for this crop in this country.")
        st.stop()

    # Crop Suitability
    crop_suitability(df_clean, country)

    # Exploratory Data Analysis
    plot_eda(filtered_df)

    # Ensure all required columns exist and have valid data
    required_columns = ['Area harvested', 'Yield', 'Production']
    if not all(col in filtered_df.columns for col in required_columns):
        st.error("❌ Some required columns are missing.")
        st.stop()

    # Remove rows with zero values for modeling
    filtered_df = filtered_df[(filtered_df['Area harvested'] > 0) & 
                              (filtered_df['Yield'] > 0) & 
                              (filtered_df['Production'] > 0)]

    if len(filtered_df) < 5:
        st.warning("⚠️ Not enough valid non-zero data points for training models.")
        st.stop()

    # Prepare data for modeling
    X = filtered_df[['Area harvested', 'Yield', 'Year']]
    y = filtered_df['Production']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model_results = train_model(X_train, X_test, y_train, y_test)

    # Display Model Performance
    st.subheader("📈 Model Performance")
    results_df = pd.DataFrame([{
        'Model': 'Linear Regression',
        'MSE': model_results['MSE'],
        'R²': model_results['R²'],
        'MAE': model_results['MAE']
    }])
    st.table(results_df)

    # Prediction Interface
    prediction_interface(model_results['model'])

if __name__ == "__main__":
    main()