import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from datetime import timedelta
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Agricultural Commodity Forecasting",
    page_icon="🌾",
    layout="wide"
)

# Title and description
st.title("🌾 Agricultural Commodity Arrival Forecasting")
st.markdown("""
This application predicts future agricultural commodity arrivals at various APMCs 
(Agricultural Produce Market Committees) based on historical data and market trends.
""")

# [Keep all the previous functions (clean_and_validate_data, calculate_base_features, etc.) the same]

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        df = pd.read_csv('Monthly_data_cmo.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def save_model(model, metrics):
    """Save the trained model and metrics"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_data = {
        'model': model,
        'metrics': metrics
    }
    
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

def load_model():
    """Load the trained model and metrics"""
    try:
        with open('models/trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['metrics']
    except:
        return None, None

def train_model_with_progress():
    """Train model with progress bar"""
    df = load_data()
    if df is None:
        return None, None
    
    with st.spinner("Training model... This may take a few minutes."):
        progress_bar = st.progress(0)
        
        # Update progress bar at key points
        progress_bar.progress(20)
        df_prepared = prepare_data(df)
        
        progress_bar.progress(40)
        # Create model and train
        model, metrics = train_and_evaluate_model(df_prepared)
        
        # Save the trained model
        save_model(model, metrics)
        
        progress_bar.progress(100)
        st.success("Model training completed!")
        
        return model, metrics

def main():
    # Sidebar
    st.sidebar.header("Model Training")
    
    # Check if model exists
    model, metrics = load_model()
    
    if model is None:
        st.warning("No trained model found. Please train the model first.")
    else:
        st.success("Model loaded successfully!")
        # Display metrics
        st.sidebar.subheader("Model Performance Metrics")
        for metric, value in metrics.items():
            st.sidebar.metric(metric, f"{value:.2f}")
    
    if st.sidebar.button("Train New Model"):
        model, metrics = train_model_with_progress()
    
    # Load data
    df = load_data()
    if df is None or model is None:
        st.warning("Please ensure both data and model are available.")
        return
    
    # Get unique values for dropdowns
    commodities = sorted(df['Commodity'].unique())
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_commodity = st.selectbox("Select Commodity", commodities)
    
    with col2:
        apmcs = sorted(df[df['Commodity'] == selected_commodity]['APMC'].unique())
        selected_apmc = st.selectbox("Select APMC", apmcs)
    
    with col3:
        months = st.slider("Number of months to predict", 1, 12, 3)
    
    # Make prediction when user clicks
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            try:
                predictions = predict_future(
                    model,
                    df,
                    selected_commodity,
                    selected_apmc,
                    months
                )
                
                # Display predictions
                st.subheader("Forecast Results")
                
                # Create two columns for table and chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(predictions.style.format({
                        'Predicted_Arrivals': '{:,.2f}'
                    }))
                
                with col2:
                    # Create line chart
                    fig = px.line(
                        predictions,
                        x='Date',
                        y='Predicted_Arrivals',
                        title=f'Predicted Arrivals for {selected_commodity} at {selected_apmc}'
                    )
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Predicted Arrivals (Quintals)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Historical data visualization
                st.subheader("Historical Data")
                historical_data = df[
                    (df['Commodity'] == selected_commodity) & 
                    (df['APMC'] == selected_apmc)
                ].copy()
                historical_data['date'] = pd.to_datetime(historical_data['date'])
                
                hist_fig = px.line(
                    historical_data,
                    x='date',
                    y='arrivals_in_qtl',
                    title=f'Historical Arrivals for {selected_commodity} at {selected_apmc}'
                )
                hist_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Actual Arrivals (Quintals)"
                )
                st.plotly_chart(hist_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")

if __name__ == "__main__":
    main()
