from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import time

from ibb_transportation_forecast.preprocessor import DataPreprocessor
from .utils import load_config

class MetroForecastModel:
    def __init__(self):
        self.config = load_config()
        self.models = self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all model types with their parameters"""
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from xgboost import XGBRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from lightgbm import LGBMRegressor
        
        return {
            'LinearRegression': LinearRegression(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(n_jobs=-1),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'XGBRegressor': XGBRegressor(n_jobs=-1),
            'SVR': SVR(kernel='rbf'),
            'KNeighborsRegressor': KNeighborsRegressor(n_jobs=-1),
            'MLPRegressor': MLPRegressor(),
            'LGBMRegressor': LGBMRegressor(n_jobs=-1)
        }
    
    def get_model(self, model_name):
        """Get initialized model by name"""
        return self.models.get(model_name)
    
    def train_model(self, model_name, X_train, y_train):
        """Train model with optional hyperparameter tuning"""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Invalid model name: {model_name}")
            
        preprocessor = DataPreprocessor().create_preprocessor(model_name)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train with hyperparameter tuning if specified
        param_grid = self._get_param_grid(model_name)
        if param_grid:
            search = RandomizedSearchCV(
                pipeline, 
                param_distributions=param_grid,
                cv=3, 
                n_iter=5, 
                n_jobs=-1, 
                verbose=1, 
                random_state=self.config['model_settings']['random_state']
            )
            search.fit(X_train, y_train)
            return search.best_estimator_
        
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def _get_param_grid(self, model_name):
        """Get hyperparameter grid for specific models"""
        param_grids = {
            'RandomForestRegressor': {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [None, 10]
            },
            'XGBRegressor': {
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.01, 0.1]
            },
            'LGBMRegressor': {
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.01, 0.1]
            }
        }
        return param_grids.get(model_name)
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    
    def forecast(self, model_name, station_data):
        """Run full forecasting pipeline for a station"""
        start_time = time.time()
        
        # Get last N days for holdout set
        last_dates = station_data['date'].drop_duplicates().sort_values().tail(
            self.config['model_settings']['test_size']
        )
        
        # Split data
        holdout = station_data[station_data['date'].isin(last_dates)].copy()
        train = station_data[~station_data['date'].isin(last_dates)].copy()
        
        # Prepare features
        features = (
            self.config['features']['numerical'] + 
            self.config['features']['categorical']['dependent'] + 
            self.config['features']['categorical']['independent']
        )
        X_train = train[features]
        y_train = train['number_of_passage']
        X_test = holdout[features]
        y_test = holdout['number_of_passage']
        
        # Train model
        model = self.train_model(model_name, X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test)
        metrics['duration'] = time.time() - start_time
        # Every value in metrics must be of type float
        metrics = {key: float(value) for key, value in metrics.items()}

        
        # Generate predictions
        holdout['predicted'] = model.predict(X_test)
        
        # Aggregate results by date
        forecast_table = (holdout.groupby(['date'])['predicted']
                        .sum().clip(lower=0).round().astype(int))
        actual_table = (holdout.groupby(['date'])['number_of_passage']
                        .sum())
        
        return forecast_table, actual_table, metrics