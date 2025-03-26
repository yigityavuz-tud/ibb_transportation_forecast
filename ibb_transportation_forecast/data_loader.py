import boto3
import pandas as pd
import io
import yaml
from datetime import datetime, timedelta
import requests
from typing import Dict, Tuple, List, Optional

def load_config() -> Dict:
    """Load configuration from YAML file"""
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

class DataLoader:
    def __init__(self):
        self.config = load_config()
        self.s3 = boto3.client('s3')
        
    def load_transport_data(self) -> pd.DataFrame:
        """Load main transportation data from S3"""
        obj = self.s3.get_object(
            Bucket=self.config['s3']['bucket_name'],
            Key=self.config['s3']['files']['transportation']
        )
        df = pd.read_csv(obj['Body'])
        print(f"Loaded transportation data with {len(df)} records")
        return df
    
    def load_weather_data(self, from_api: bool = False, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load weather data either from S3 or API
        Args:
            from_api: If True, fetch from API instead of S3
            start_date: Start date for API fetch
            end_date: End date for API fetch
        """
        if from_api and start_date and end_date:
            print("Fetching weather data from API...")
            return self._fetch_weather_data(start_date, end_date)
        
        print("Loading weather data from S3...")
        obj = self.s3.get_object(
            Bucket=self.config['s3']['bucket_name'],
            Key=self.config['s3']['files']['weather']
        )
        weather_df = pd.read_csv(obj['Body'])
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
        return weather_df[['datetime', 'precip', 'wind_spd', 'temp', 'rh', 'clouds', 'snow']]
    
    def load_bank_holidays(self) -> pd.DataFrame:
        """Load bank holidays data"""
        df = self._load_supporting_data('bank_holidays')
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} bank holidays")
        return df
    
    def load_football_schedule(self) -> pd.DataFrame:
        """Load football match schedule data"""
        df = self._load_supporting_data('bigthree_schedule')
        df['date'] = pd.to_datetime(df['date'])
        
        # Clean team names
        df['home'] = (df['home'].str.lower()
                      .str.replace('ı', 'i').str.replace('ş', 's')
                      .str.replace('ğ', 'g').str.replace('ç', 'c')
                      .str.replace('ü', 'u').str.replace('ö', 'o'))
        df['away'] = (df['away'].str.lower()
                      .str.replace('ı', 'i').str.replace('ş', 's')
                      .str.replace('ğ', 'g').str.replace('ç', 'c')
                      .str.replace('ü', 'u').str.replace('ö', 'o'))
        
        print(f"Loaded {len(df)} football matches")
        return df
    
    def load_school_terms(self) -> pd.DataFrame:
        """Load school term dates"""
        df = self._load_supporting_data('school_terms')
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        print(f"Loaded {len(df)} school terms")
        return df
    
    def load_ramadan_dates(self) -> pd.DataFrame:
        """Load Ramadan dates"""
        df = self._load_supporting_data('ramadan')
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} Ramadan dates")
        return df
    
    def _load_supporting_data(self, data_type: str) -> pd.DataFrame:
        """Internal method to load supporting data files"""
        obj = self.s3.get_object(
            Bucket=self.config['s3']['bucket_name'],
            Key=self.config['s3']['files'][data_type]
        )
        
        if self.config['s3']['files'][data_type].endswith('.csv'):
            return pd.read_csv(io.BytesIO(obj['Body'].read()))
        return pd.read_excel(io.BytesIO(obj['Body'].read()))
    
    def _fetch_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch weather data from API"""
        date_ranges = self._split_date_range(start_date, end_date)
        all_weather_data = []
        
        for start, end in date_ranges:
            params = {
                "lat": self.config['api']['weatherbit']['lat'],
                "lon": self.config['api']['weatherbit']['lon'],
                "start_date": start.strftime('%Y-%m-%d'),
                "end_date": end.strftime('%Y-%m-%d'),
                "key": self.config['api']['weatherbit']['api_key']
            }
            response = requests.get(self.config['api']['weatherbit']['url'], params=params)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['data'])
                all_weather_data.append(df)
        
        return pd.concat(all_weather_data, ignore_index=True)
    
    def _split_date_range(self, start_date: datetime, end_date: datetime, 
                         max_months: int = 7) -> List[Tuple[datetime, datetime]]:
        """Split date range into chunks for API calls"""
        date_ranges = []
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + timedelta(days=30 * max_months)
            if current_end > end_date:
                current_end = end_date
            date_ranges.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)
        return date_ranges