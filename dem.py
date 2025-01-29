import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import BytesIO

# Configure the Streamlit page
st.set_page_config(
    page_title="AgriPredict | Commodity Price & Arrival Forecasting",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .header-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

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
@st.cache_data
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
@st.cache_data
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
@st.cache_data
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
@st.cache_data
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
@st.cache_data
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
@st.cache_data
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
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'RMSE': rmse,
            'R2': r2,
            'RMSE_percentage': (rmse / y_test.mean()) * 100,
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
        price_data = df_commodity[['date', 'modal_price']].rename(
                columns={'date': 'ds', 'modal_price': 'y'}
            )
            
        prophet_model = Prophet()
        prophet_model.fit(price_data)
            
            # Generate price predictions
        future_prices = prophet_model.make_future_dataframe(periods=months, freq='M')
        price_forecast = prophet_model.predict(future_prices)
        predicted_prices = price_forecast['yhat'].tail(months).values
            
            # Combine predictions
        result = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Arrivals': predictions,
                'Predicted_Prices': predicted_prices
            })
            
        return result
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        raise
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Monthly_data_cmo.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
@st.cache_data
def create_plotly_chart(predictions, title, y_label):
    fig = go.Figure()
    
    # Add the prediction line
    fig.add_trace(go.Scatter(
        x=predictions['Date'],
        y=predictions['Predicted_Arrivals' if 'Predicted_Arrivals' in predictions else 'Predicted_Prices'],
        mode='lines+markers',
        name='Prediction',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Date",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig
@st.cache_data
def download_predictions(predictions):
    output = BytesIO()
    predictions.to_csv(output, index=False)
    b64 = base64.b64encode(output.getvalue()).decode()
    return f'data:application/octet-stream;base64,{b64}'

def main():
    # Header
    
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 style='text-align: center;'>🌾 AgriPredict</h1>
            <h3 style='text-align: center;'>Agricultural Commodity Price & Arrival Forecasting</h3>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()
   
    if df is not None:
        # Sidebar
        st.sidebar.header("📊 Forecast Parameters")
        
        # Get unique values for dropdowns
        commodities = sorted(df['Commodity'].unique())
        apmcs = sorted(df['APMC'].unique())
        
        # User inputs
        commodity = st.sidebar.selectbox("Select Commodity", commodities)
        apmc = st.sidebar.selectbox("Select APMC", apmcs)
        months = st.sidebar.slider("Forecast Period (Months)", 1, 12, 3)
        
        # Generate forecast button
        if st.sidebar.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    # Train the model
                    model, metrics = train_and_evaluate_model(df)

                    # Get predictions
                    predictions = predict_future(model, df, commodity, apmc, months)
                    
                    # Create two columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                            <div class="metric-card">
                                <h4>Average Predicted Arrivals</h4>
                                <h2>{:,.2f} qtl</h2>
                            </div>
                        """.format(predictions['Predicted_Arrivals'].mean()), 
                        unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                            <div class="metric-card">
                                <h4>Average Predicted Price</h4>
                                <h2>₹{:,.2f}</h2>
                            </div>
                        """.format(predictions['Predicted_Prices'].mean()),
                        unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("""
                            <div class="metric-card">
                                <h4>Forecast Period</h4>
                                <h2>{} Months</h2>
                            </div>
                        """.format(months),
                        unsafe_allow_html=True)
                    
                    # Charts
                    st.plotly_chart(
                        create_plotly_chart(
                            predictions,
                            f"Predicted Arrivals for {commodity} at {apmc}",
                            "Arrivals (qtl)"
                        ),
                        use_container_width=True
                    )
                    
                    st.plotly_chart(
                        create_plotly_chart(
                            predictions,
                            f"Predicted Prices for {commodity} at {apmc}",
                            "Price (₹)"
                        ),
                        use_container_width=True
                    )
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Predictions",
                        data=download_predictions(predictions),
                        file_name=f"predictions_{commodity}_{apmc}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
        
        # Historical data visualization
        st.sidebar.markdown("---")
        st.sidebar.header("📈 Historical Data")
        if st.sidebar.checkbox("Show Historical Data"):
            filtered_df = df[
                (df['Commodity'] == commodity) & 
                (df['APMC'] == apmc)
            ].sort_values('date')
            
            st.subheader("Historical Data Analysis")
            
            # Historical trends
            fig = px.line(
                filtered_df,
                x='date',
                y=['min_price', 'max_price', 'modal_price'],
                title=f'Historical Price Trends for {commodity} at {apmc}',
                labels={'value': 'Price (₹)', 'date': 'Date'},
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal patterns
            monthly_avg = filtered_df.groupby('date').agg({'modal_price': 'mean'}).reset_index()
            seasonal_fig = px.line(
                monthly_avg,
                x='date',
                y='modal_price',
                title=f"Seasonal Price Patterns for {commodity} at {apmc}",
                labels={'modal_price': 'Average Price (₹)', 'date': 'Month'},
                template="plotly_white"
            )
            st.plotly_chart(seasonal_fig, use_container_width=True)


if __name__ == "__main__":
    main()
