import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import calendar

# Generate synthetic weather data for 5 years
def generate_weather_data(years=5, city="Sample City"):
    np.random.seed(42)
    
    # Date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365 * years)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base temperature by month (Northern Hemisphere pattern)
    monthly_base_temp = {
        1: 0, 2: 2, 3: 7, 4: 12, 5: 18, 6: 23,
        7: 25, 8: 24, 9: 20, 10: 15, 11: 8, 12: 2
    }
    
    # Generate temperatures with seasonal patterns and random variations
    temperatures = []
    for date in dates:
        month = date.month
        base_temp = monthly_base_temp[month]
        
        # Add yearly trend (slight warming)
        years_from_start = (date - start_date).days / 365
        warming_trend = years_from_start * 0.2  # 0.2°C per year
        
        # Add random variation
        daily_variation = np.random.normal(0, 3)  # Daily random fluctuation
        
        # Combine components
        temp = base_temp + warming_trend + daily_variation
        temperatures.append(temp)
    
    # Precipitation patterns (more in spring/fall, less in summer/winter)
    precipitation = []
    for date in dates:
        month = date.month
        
        # Base precipitation probability
        if month in [4, 5, 9, 10]:  # Spring and Fall
            precip_prob = 0.4
        elif month in [6, 7, 8]:    # Summer
            precip_prob = 0.3
        else:                        # Winter
            precip_prob = 0.35
        
        # Determine if it rained
        if np.random.random() < precip_prob:
            # Amount of precipitation (in mm)
            if month in [12, 1, 2]:  # Winter - potentially more precipitation
                amount = np.random.gamma(2, 4)
            else:
                amount = np.random.gamma(2, 3)
        else:
            amount = 0
        
        precipitation.append(amount)
    
    # Humidity levels (correlated with precipitation and temperature)
    humidity = []
    for i, date in enumerate(dates):
        base_humidity = 60
        
        # Higher humidity when precipitation
        precip_effect = min(precipitation[i] * 2, 25)
        
        # Lower humidity when hot
        temp_effect = max(-15, min(0, 15 - temperatures[i]))
        
        # Random variation
        random_effect = np.random.normal(0, 5)
        
        # Combine effects and clip to valid range
        humid = base_humidity + precip_effect + temp_effect + random_effect
        humid = max(30, min(100, humid))
        
        humidity.append(humid)
    
    # Wind speed
    wind_speed = np.random.gamma(2, 1.5, size=len(dates))
    
    # Create DataFrame
    weather_data = pd.DataFrame({
        'Date': dates,
        'Temperature': temperatures,
        'Precipitation': precipitation,
        'Humidity': humidity,
        'WindSpeed': wind_speed,
        'City': city
    })
    
    # Add some extreme weather events
    # Heat waves
    for year in range(years):
        # Summer heat wave
        heat_wave_start = start_date + timedelta(days=365 * year + 180 + np.random.randint(-20, 20))
        heat_wave_duration = np.random.randint(3, 8)
        
        for i in range(heat_wave_duration):
            date = heat_wave_start + timedelta(days=i)
            if date in weather_data['Date'].values:
                idx = weather_data[weather_data['Date'] == date].index
                weather_data.loc[idx, 'Temperature'] += np.random.uniform(5, 10)
                weather_data.loc[idx, 'Humidity'] -= np.random.uniform(5, 15)
    
    # Cold snaps
    for year in range(years):
        # Winter cold snap
        cold_snap_start = start_date + timedelta(days=365 * year + 30 + np.random.randint(-20, 20))
        cold_snap_duration = np.random.randint(3, 6)
        
        for i in range(cold_snap_duration):
            date = cold_snap_start + timedelta(days=i)
            if date in weather_data['Date'].values:
                idx = weather_data[weather_data['Date'] == date].index
                weather_data.loc[idx, 'Temperature'] -= np.random.uniform(5, 12)
    
    # Heavy rain events
    for year in range(years):
        # Random heavy rain events
        for _ in range(3):
            heavy_rain_date = start_date + timedelta(days=np.random.randint(0, 365) + 365 * year)
            if heavy_rain_date in weather_data['Date'].values:
                idx = weather_data[weather_data['Date'] == heavy_rain_date].index
                weather_data.loc[idx, 'Precipitation'] = np.random.uniform(20, 40)
                weather_data.loc[idx, 'Humidity'] = min(100, weather_data.loc[idx, 'Humidity'] + np.random.uniform(10, 20))
    
    # Add derived features
    weather_data['Year'] = weather_data['Date'].dt.year
    weather_data['Month'] = weather_data['Date'].dt.month
    weather_data['Day'] = weather_data['Date'].dt.day
    weather_data['Season'] = weather_data['Month'].apply(get_season)
    weather_data['MonthName'] = weather_data['Month'].apply(lambda x: calendar.month_name[x])
    weather_data['DayOfWeek'] = weather_data['Date'].dt.day_name()
    
    return weather_data

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

class WeatherAnalyzer:
    def __init__(self, data):
        self.data = data
        self.city = data['City'].iloc[0]
    
    def basic_statistics(self):
        """Calculate basic statistics for weather variables."""
        print(f"Weather Analysis for {self.city}")
        print("=" * 50)
        
        # Overall statistics
        stats = self.data[['Temperature', 'Precipitation', 'Humidity', 'WindSpeed']].describe()
        print("Overall Weather Statistics:")
        print(stats)
        print("-" * 50)
        
        # Statistics by season
        seasonal_stats = self.data.groupby('Season')[['Temperature', 'Precipitation', 'Humidity', 'WindSpeed']].agg(
            ['mean', 'std', 'min', 'max']
        )
        print("\nSeasonal Weather Statistics:")
        print(seasonal_stats)
        print("-" * 50)
        
        # Statistics by month
        monthly_stats = self.data.groupby('MonthName')[['Temperature', 'Precipitation', 'Humidity']].agg(
            ['mean', 'std', 'min', 'max']
        )
        # Sort by month number
        month_order = [calendar.month_name[i] for i in range(1, 13)]
        monthly_stats = monthly_stats.reindex(month_order)
        
        print("\nMonthly Temperature Statistics:")
        print(monthly_stats['Temperature'])
        print("-" * 50)
        
        return stats, seasonal_stats, monthly_stats
    
    def plot_temperature_trends(self):
        """Plot temperature trends over time."""
        # Calculate monthly averages
        monthly_avg = self.data.groupby(['Year', 'Month'])['Temperature'].mean().reset_index()
        monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(Day=1))
        
        plt.figure(figsize=(14, 6))
        plt.plot(monthly_avg['Date'], monthly_avg['Temperature'], 'o-', color='tab:red')
        
        # Add trend line
        x = (monthly_avg['Date'] - monthly_avg['Date'].min()).dt.days.values
        y = monthly_avg['Temperature'].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(monthly_avg['Date'], p(x), 'r--', linewidth=2)
        
        # Calculate trend
        trend_per_year = z[0] * 365  # degrees per year
        
        plt.title(f'Monthly Average Temperature ({self.city})\nTrend: {trend_per_year:.2f}°C per year')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot by season
        seasonal_temp = self.data.groupby(['Year', 'Season'])['Temperature'].mean().reset_index()
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        
        plt.figure(figsize=(14, 8))
        
        for season in seasons:
            season_data = seasonal_temp[seasonal_temp['Season'] == season]
            plt.plot(season_data['Year'], season_data['Temperature'], 'o-', label=season)
        
        plt.title(f'Seasonal Temperature Trends ({self.city})')
        plt.xlabel('Year')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return trend_per_year
    
    def plot_seasonal_patterns(self):
        """Plot seasonal patterns of temperature and precipitation."""
        # Monthly patterns
        monthly_data = self.data.groupby('MonthName')[['Temperature', 'Precipitation', 'Humidity']].mean()
        month_order = [calendar.month_name[i] for i in range(1, 13)]
        monthly_data = monthly_data.reindex(month_order)
        
        # Temperature by month
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=monthly_data.index, y=monthly_data['Temperature'], palette='YlOrRd')
        
        plt.title(f'Average Temperature by Month ({self.city})')
        plt.xlabel('Month')
        plt.ylabel('Temperature (°C)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Precipitation by month
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=monthly_data.index, y=monthly_data['Precipitation'], palette='Blues')
        
        plt.title(f'Average Precipitation by Month ({self.city})')
        plt.xlabel('Month')
        plt.ylabel('Precipitation (mm)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Boxplot of temperature by season
        plt.figure(figsize=(12, 6))
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        sns.boxplot(x='Season', y='Temperature', data=self.data, order=season_order, palette='viridis')
        
        plt.title(f'Temperature Distribution by Season ({self.city})')
        plt.xlabel('Season')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return monthly_data
    
    def analyze_precipitation(self):
        """Analyze precipitation patterns."""
        # Calculate rainy days by month
        self.data['IsRainy'] = self.data['Precipitation'] > 0.1
        rainy_days = self.data.groupby('MonthName')['IsRainy'].mean() * 100  # percentage
        month_order = [calendar.month_name[i] for i in range(1, 13)]
        rainy_days = rainy_days.reindex(month_order)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=rainy_days.index, y=rainy_days.values, palette='Blues')
        
        plt.title(f'Percentage of Rainy Days by Month ({self.city})')
        plt.xlabel('Month')
        plt.ylabel('Rainy Days (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Analyze heavy rainfall events
        heavy_rain = self.data[self.data['Precipitation'] > 10]
        heavy_rain_by_month = heavy_rain.groupby('MonthName').size()
        heavy_rain_by_month = heavy_rain_by_month.reindex(month_order, fill_value=0)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=heavy_rain_by_month.index, y=heavy_rain_by_month.values, palette='Blues')
        
        plt.title(f'Heavy Rainfall Events by Month ({self.city})')
        plt.xlabel('Month')
        plt.ylabel('Number of Events (Precipitation > 10mm)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Distribution of precipitation amounts
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data[self.data['Precipitation'] > 0]['Precipitation'], 
                    bins=20, kde=True, color='blue')
        