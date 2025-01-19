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

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Agricultural Commodity Forecasting",
    page_icon="🌾",
    layout="wide"
)

# Title and description
st.title("🌾 Agricultural Commodity Price Forecasting")
st.markdown("""
This application predicts future agricultural commodity arrivals at various APMCs 
(Agricultural Produce Market Committees) based on historical data and market trends.
""")

# [Keep all the preprocessing functions the same: clean_and_validate_data, calculate_base_features, etc.]

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        # Updated path to load directly from root
        df = pd.read_csv('Monthly_data_cmo.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
def clean_and_validate_data(df):
    """Clean and validate the data"""
    df = df.copy()
    
    # Replace 0 prices with NaN
    price_columns = ['min_price', 'max_price', 'modal_price']
    df[price_columns] = df[price_columns].replace(0, np.nan)
    
    # Remove rows with invalid prices
    df = df.dropna(subset=price_columns)
    
    # Ensure prices make logical sense
    df = df[df['max_price'] >= df['min_price']]
    df = df[df['modal_price'] >= df['min_price']]
    df = df[df['modal_price'] <= df['max_price']]
    
    return df

def calculate_base_features(df):
    """Calculate initial set of features"""
    df = df.copy()
    
    # Convert date and extract time features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    
    # Calculate price-based features
    df['price_range'] = df['max_price'] - df['min_price']
    df['price_volatility'] = (df['price_range'] / df['modal_price']).clip(0, 1)
    
    return df

def add_seasonal_features(df):
    """Add seasonal decomposition features"""
    df = df.copy()
    
    # Season indicators (Indian agricultural seasons)
    df['is_kharif'] = df['month_num'].isin([6, 7, 8, 9, 10])  # Kharif season
    df['is_rabi'] = df['month_num'].isin([11, 12, 1, 2, 3])   # Rabi season
    df['is_zaid'] = df['month_num'].isin([4, 5])              # Zaid season
    
    # Cyclical encoding of months
    df['month_sin'] = np.sin(2 * np.pi * df['month_num']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num']/12)
    
    return df

def add_market_features(df):
    """Add market-related features"""
    df = df.copy()
    groups = df.groupby(['APMC', 'Commodity'])
    
    # Rolling statistics (3-month window)
    df['rolling_mean_price'] = groups['modal_price'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['rolling_std_price'] = groups['modal_price'].transform(lambda x: x.rolling(3, min_periods=1).std())
    df['rolling_mean_arrivals'] = groups['arrivals_in_qtl'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    # Market concentration
    df['market_size'] = groups['arrivals_in_qtl'].transform('mean')
    df['relative_market_size'] = df['arrivals_in_qtl'] / df['market_size']
    
    # Price trends
    df['price_momentum'] = groups['modal_price'].pct_change()
    
    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    return df

def add_lag_features(df):
    """Add lagged features"""
    df = df.copy()
    groups = df.groupby(['APMC', 'Commodity'])
    
    # Create multiple lag features
    for lag in [1, 2, 3]:
        df[f'arrivals_lag_{lag}'] = groups['arrivals_in_qtl'].shift(lag)
        df[f'price_lag_{lag}'] = groups['modal_price'].shift(lag)
    
    # Fill missing values with group medians
    for col in df.columns[df.columns.str.contains('lag')]:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def prepare_data(df):
    """Complete data preparation pipeline"""
    print("Cleaning data...")
    df = clean_and_validate_data(df)
    
    print("Calculating base features...")
    df = calculate_base_features(df)
    
    print("Adding seasonal features...")
    df = add_seasonal_features(df)
    
    print("Adding market features...")
    df = add_market_features(df)
    
    print("Adding lag features...")
    df = add_lag_features(df)
    
    # Final cleaning of any remaining NaN values
    df = df.dropna()
    
    return df

def create_forecast_model():
    """Create enhanced forecasting model"""
    categorical_features = ['APMC', 'Commodity', 'state_name', 'district_name']
    numerical_features = [
        'year', 'month_num', 'min_price', 'max_price', 'modal_price',
        'price_range', 'price_volatility', 'market_size', 'relative_market_size',
        'rolling_mean_price', 'rolling_std_price', 'rolling_mean_arrivals',
        'price_momentum', 'month_sin', 'month_cos',
        'arrivals_lag_1', 'arrivals_lag_2', 'arrivals_lag_3',
        'price_lag_1', 'price_lag_2', 'price_lag_3'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])

def train_and_evaluate_model(df):
    """Train and evaluate the model"""
    try:
        # Prepare all data
        df_prepared = prepare_data(df)
        
        print(f"\nDataset shape after preparation: {df_prepared.shape}")
        print(f"Number of unique commodities: {df_prepared['Commodity'].nunique()}")
        print(f"Number of unique APMCs: {df_prepared['APMC'].nunique()}")
        
        # Create stratification column
        df_prepared['temp_strat'] = df_prepared['APMC'] + '_' + df_prepared['Commodity']
        
        # Split features and target
        X = df_prepared.drop(['arrivals_in_qtl', 'date', 'temp_strat'], axis=1)
        y = df_prepared['arrivals_in_qtl']
        
        # Perform regular split (no stratification)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("\nTraining enhanced model...")
        model = create_forecast_model()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        # r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'Test_Set_Size': len(y_test),
            'Training_Set_Size': len(y_train)
        }
        
        print("\nModel training completed successfully!")
        return model, metrics
    
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        print("\nFull error details:")
        print(traceback.format_exc())
        raise


def predict_future(model, df, commodity, apmc, months=3):
    """Predict future values for a given commodity and APMC."""
    try:
        # Prepare the base data for predictions
        df_commodity = df[(df['Commodity'] == commodity) & (df['APMC'] == apmc)].copy()

        if df_commodity.empty:
            raise ValueError(f"No data found for commodity '{commodity}' in APMC '{apmc}'.")

        # Sort by date
        df_commodity['date'] = pd.to_datetime(df_commodity['date'])
        df_commodity = df_commodity.sort_values('date')

        # Extract the latest data point
        latest_data = df_commodity.iloc[-1]

        # Generate future dates
        future_dates = [latest_data['date'] + timedelta(days=30 * i) for i in range(1, months + 1)]

        # Create a DataFrame for future predictions
        future_df = pd.DataFrame({
            'date': future_dates,
            'APMC': apmc,
            'Commodity': commodity,
            'state_name': latest_data['state_name'],
            'district_name': latest_data['district_name'],
            'min_price': latest_data['min_price'],
            'max_price': latest_data['max_price'],
            'modal_price': latest_data['modal_price'],
            'arrivals_in_qtl': latest_data['arrivals_in_qtl'],
            'year': [d.year for d in future_dates],
            'month_num': [d.month for d in future_dates]
        })

        # Add seasonal and market features to the future data
        future_df = calculate_base_features(future_df)
        future_df = add_seasonal_features(future_df)
        future_df = add_market_features(future_df)

        # Add lag features using the latest available data
        for lag in [1, 2, 3]:
            future_df[f'arrivals_lag_{lag}'] = latest_data.get(f'arrivals_lag_{lag}', 0)
            future_df[f'price_lag_{lag}'] = latest_data.get(f'price_lag_{lag}', 0)

        # Drop unnecessary columns
        future_df = future_df.drop(['date', 'arrivals_in_qtl'], axis=1, errors='ignore')

        # Ensure all expected columns exist in the prediction DataFrame
        model_columns = model.named_steps['preprocessor'].transformers_[0][2] + \
                        list(model.named_steps['preprocessor'].transformers_[1][2])
        for col in model_columns:
            if col not in future_df:
                future_df[col] = 0

        # Make predictions
        predictions = model.predict(future_df)

        # Return a DataFrame with predictions
        result = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Arrivals': predictions
        })

        return result

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
# [Previous imports and functions remain the same until train_model_with_progress]

def save_model(model, metrics):
    """Save the trained model and metrics"""
    model_data = {
        'model': model,
        'metrics': metrics
    }
    
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

def load_model():
    """Load the trained model and metrics"""
    try:
        # Updated path to load directly from root
        with open('trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['metrics']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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
    # st.sidebar.header("Model Training")
    
    # Check if model exists
    model, metrics = load_model()
    
    if model is None:
        st.warning("No trained model found. Please train the model first.")
    else:
        st.success("Model loaded successfully!")
        # Display metrics
        # st.sidebar.subheader("Model Performance Metrics")
        # for metric, value in metrics.items():
        #     st.sidebar.metric(metric, f"{value:.2f}")
    
    # if st.sidebar.button("Train New Model"):
    #     model, metrics = train_model_with_progress()
    
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
                        yaxis_title="Predicted Price"
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
