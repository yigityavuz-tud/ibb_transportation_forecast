import re
import pandas as pd
from typing import Dict, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .utils import load_config

class DataPreprocessor:
    def __init__(self):
        self.config = load_config()
        
    def clean_transport_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare transportation data
        Args:
            df: Raw transportation DataFrame
        Returns:
            Cleaned DataFrame with standardized station names
        """
        # Clean station names
        df['station'] = df['station_poi_desc_cd'].apply(
            lambda x: re.sub(r'\b(GUNEY|KUZEY|BATI|DOGU|STAD GIRISI)\b', '', x).strip()
        )
        df['station'] = df['station'].str.replace(r'\d+$', '', regex=True)
        df['station'] = df['station'].str.rstrip()
        
        # Convert and validate dates
        df['transition_date'] = pd.to_datetime(df['transition_date'])
        return df
    
    def aggregate_transport_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transportation data by date, type, and station
        Args:
            df: Cleaned transportation DataFrame
        Returns:
            Aggregated DataFrame with sum of passages
        """
        agg_df = df.groupby([
            'transition_date', 'transfer_type', 
            'transaction_type_desc', 'product_kind', 'station'
        ])['number_of_passage'].sum().reset_index()
        
        # Filter to valid product kinds
        valid_products = ['INDIRIMLI1', 'INDIRIMLI2', 'TAM', 'UCRETSIZ']
        return agg_df[agg_df['product_kind'].isin(valid_products)]
    
    def remove_transport_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from transportation data using IQR method
        Args:
            df: Aggregated transportation DataFrame
        Returns:
            DataFrame with outliers removed
        """
        pivot = df.pivot_table(
            index='transition_date', 
            columns=['product_kind'], 
            values='number_of_passage', 
            aggfunc='sum'
        )
        
        Q1 = pivot.quantile(0.25)
        Q3 = pivot.quantile(0.75)
        IQR = Q3 - Q1
        outliers = pivot[(pivot < (Q1 - 1.5 * IQR))].dropna(how='all')
        
        return df[~df['transition_date'].isin(outliers.index)].copy()
    
    def merge_supporting_data(self, transport_df: pd.DataFrame, 
                            weather_df: pd.DataFrame,
                            holidays_df: pd.DataFrame,
                            football_df: pd.DataFrame,
                            school_df: pd.DataFrame,
                            ramadan_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all data sources into a single DataFrame
        Args:
            Various DataFrames containing different data sources
        Returns:
            Combined DataFrame with all features
        """
        # Merge weather data
        combined = pd.merge(
            transport_df, 
            weather_df, 
            how='left', 
            left_on='transition_date', 
            right_on='datetime'
        ).drop(columns=['datetime'])
        
        # Add bank holidays flag
        combined = pd.merge(
            combined,
            holidays_df[['date']].assign(isBankHoliday=1),
            how='left',
            left_on='transition_date',
            right_on='date'
        ).drop(columns=['date'])
        combined['isBankHoliday'] = combined['isBankHoliday'].fillna(0)
        
        # Add football match indicators
        for team in football_df['home'].unique():
            combined[team] = combined['transition_date'].isin(
                football_df[football_df['home'] == team]['date']
            ).astype(int)
        
        # Add school terms flag
        combined['schoolsOpen'] = 0
        for _, row in school_df.iterrows():
            mask = (combined['transition_date'] >= row['start_date']) & \
                   (combined['transition_date'] <= row['end_date'])
            combined.loc[mask, 'schoolsOpen'] = 1
        
        # Add Ramadan flag
        combined['isRamadan'] = combined['transition_date'].isin(
            ramadan_df['date']
        ).astype(int)
        
        # Rename columns and add temporal features
        combined = combined.rename(columns={
            'transition_date': 'date',
            'temp': 'temperature',
            'rh': 'relative_humidity'
        })
        
        return self.add_temporal_features(combined)
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to DataFrame
        Args:
            df: Combined DataFrame
        Returns:
            DataFrame with added temporal features
        """
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Convert categorical columns
        cat_cols = (
            self.config['features']['categorical']['dependent'] + 
            self.config['features']['categorical']['independent']
        )
        df[cat_cols] = df[cat_cols].astype('category')
        
        return df
    
    def create_preprocessor(self, model_type: str) -> ColumnTransformer:
        """
        Create appropriate preprocessor based on model type
        Args:
            model_type: Type of model being used
        Returns:
            Configured ColumnTransformer
        """
        numerical_features = self.config['features']['numerical']
        categorical_features = (
            self.config['features']['categorical']['dependent'] + 
            self.config['features']['categorical']['independent']
        )
        
        # Models that need feature scaling
        if model_type in ['LinearRegression', 'SVR', 'KNeighborsRegressor', 'MLPRegressor']:
            return ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
        
        # Tree-based models that don't need scaling
        return ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
    
    def prepare_station_data(self, combined_df: pd.DataFrame, 
                           station_name: str) -> pd.DataFrame:
        """
        Prepare data for a specific station
        Args:
            combined_df: Combined DataFrame with all stations
            station_name: Name of station to prepare data for
        Returns:
            DataFrame prepared for modeling for specific station
        """
        station_data = combined_df[combined_df['station'] == station_name].copy()
        # station_data.drop(columns=['station'], inplace=True)
        return station_data