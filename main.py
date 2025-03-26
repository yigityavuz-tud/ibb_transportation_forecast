#!/usr/bin/env python3
"""
Main script for Istanbul Metro Passenger Forecasting System
"""
from datetime import datetime
import pandas as pd
from ibb_transportation_forecast.data_loader import DataLoader
from ibb_transportation_forecast.preprocessor import DataPreprocessor
from ibb_transportation_forecast.model import MetroForecastModel
from ibb_transportation_forecast.utils import plot_results, load_config

def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare all required data sources"""
    print("\n=== Loading Data ===")
    loader = DataLoader()
    
    # Load core datasets
    transport_df = loader.load_transport_data()
    weather_df = loader.load_weather_data()
    
    # Load supporting data
    holidays_df = loader.load_bank_holidays()
    football_df = loader.load_football_schedule()
    school_df = loader.load_school_terms()
    ramadan_df = loader.load_ramadan_dates()
    
    print("\n=== Preprocessing Data ===")
    preprocessor = DataPreprocessor()
    
    # Clean and prepare transport data
    transport_clean = preprocessor.clean_transport_data(transport_df)
    transport_agg = preprocessor.aggregate_transport_data(transport_clean)
    transport_final = preprocessor.remove_transport_outliers(transport_agg)
    
    # Merge all data sources
    combined_data = preprocessor.merge_supporting_data(
        transport_final, weather_df, holidays_df, 
        football_df, school_df, ramadan_df
    )
    
    print(f"\nFinal combined dataset shape: {combined_data.shape}")
    return combined_data

def run_forecast_for_station(data: pd.DataFrame, station: str, model_name: str) -> dict:
    """Run forecasting pipeline for a specific station"""
    print(f"\n=== Running Forecast for {station} ===")
    preprocessor = DataPreprocessor()
    model = MetroForecastModel()
    
    # Prepare station-specific data
    station_data = preprocessor.prepare_station_data(data, station)
    
    # Run forecasting
    forecast, actual, metrics = model.forecast(model_name, station_data)
    
    # Display results
    print("\nModel Performance:")
    print(f"- MAE: {metrics['MAE']:.2f}")
    print(f"- MSE: {metrics['MSE']:.2f}")
    print(f"- R²: {metrics['R2']:.2f}")
    print(f"- Duration: {metrics['duration']:.2f} seconds")
    
    # Plot results
    plot_results(station, model_name, forecast, actual)
    
    return {
        'station': station,
        'model': model_name,
        'forecast': forecast,
        'actual': actual,
        'metrics': metrics
    }


def main():
    """Main execution function"""
    config = load_config()
    
    # Load and prepare data
    try:
        combined_data = load_and_prepare_data()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Get configuration parameters
    stations_to_run = config['model_settings'].get('stations')
    models_to_run = config['model_settings']['models']
    
    # Run forecasts
    results = []
    for station in stations_to_run:
        for model_name in models_to_run:
            try:
                result = run_forecast_for_station(combined_data, station, model_name)
                results.append(result)
            except Exception as e:
                print(f"Error processing {station} with {model_name}: {str(e)}")
                continue
    
    # Optionally save results
    if results:
        pd.DataFrame([r['metrics'] for r in results]).to_csv('forecast_results.csv', index=False)
        print("\nSaved results to forecast_results.csv")

if __name__ == "__main__":
    print("=== Istanbul Metro Passenger Forecasting ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print("\nForecasting completed!")