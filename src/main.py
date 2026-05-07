from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI(title="Sales Forecasting API")

# Load the summary we created in Colab
# This tells the API which model file belongs to which state
SUMMARY_PATH = 'models/best_models_summary.csv'

@app.get("/")
def home():
    return {"message": "Sales Forecasting API is Running"}

@app.get("/forecast/{state}")
def get_forecast(state: str):
    # 1. Check if we have a model for this state
    summary = pd.read_csv(SUMMARY_PATH)
    state_row = summary[summary['State'].str.lower() == state.lower()]
    
    if state_row.empty:
        raise HTTPException(status_code=404, detail="State not found")
    
    # 2. Load the specific model file
    model_path = f"models/{state_row.iloc[0]['State']}_model.pkl"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model file missing on server")
    
    model = joblib.load(model_path)
    
    # 3. Generate Forecast (Simplified for 8 weeks)
    # Note: For Prophet/ARIMA/XGBoost, the prediction call differs.
    # Here is a generic structure:
    best_model_type = state_row.iloc[0]['Best_Model']
    
    if best_model_type == 'ARIMA':
        forecast = model.forecast(steps=8).tolist()
    elif best_model_type == 'Prophet':
        # Prophet needs a future dataframe
        future = model.make_future_dataframe(periods=8, freq='W')
        forecast = model.predict(future)['yhat'].iloc[-8:].tolist()
    else:
    # 1. Load the processed data to get the latest features
        df_final = pd.read_csv('data/final_processed_data.csv')
        last_row = df_final[df_final['State'].str.lower() == state.lower()].iloc[-1:]
        
        forecast = []
        current_features = last_row[['lag_1', 'lag_7', 'lag_30', 'rolling_mean_4', 'month', 'week_of_year']].copy()
        
        # 2. Predict 8 weeks recursively
        for i in range(8):
            pred = model.predict(current_features)[0]
            forecast.append(float(pred))
            
            # Update features for the next week (simplified)
            current_features['lag_1'] = pred 
            # Note: In a perfect world, you'd update rolling_mean_4 too!
        
    return {
        "state": state,
        "best_model_used": "XGBoost",
        "8_week_forecast": forecast
    }