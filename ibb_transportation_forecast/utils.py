import yaml
import pandas as pd

def load_config():
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def plot_results(station_name, model_name, forecast, actual):
    """Plot forecast vs actual results"""
    import plotly.express as px
    
    plot_df = pd.DataFrame({
        'Date': forecast.index,
        'Forecasted': forecast.values,
        'Actual': actual.values
    })
    
    fig = px.line(plot_df, x='Date', y=['Forecasted', 'Actual'], 
                 title=f'Actual vs Forecasted Passages by {model_name} at {station_name}',
                 labels={'value': 'Number of Passages', 'variable': 'Legend'})
    fig.show()