import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load COVID-19 data
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
covid_data = pd.read_csv(url)

# Clean and prepare data
covid_data['date'] = pd.to_datetime(covid_data['date'])
countries_of_interest = ['United States', 'India', 'Brazil', 'United Kingdom', 'Russia']
filtered_data = covid_data[covid_data['location'].isin(countries_of_interest)]

# Analyze cases over time
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = filtered_data[filtered_data['location'] == country]
    plt.plot(country_data['date'], country_data['new_cases_smoothed'], label=country)

plt.title('COVID-19 New Cases (7-day average)')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Calculate statistics
latest_data = filtered_data.groupby('location').last().reset_index()
summary_stats = latest_data[['location', 'total_cases', 'total_deaths', 'total_cases_per_million']]
summary_stats['mortality_rate'] = (latest_data['total_deaths'] / latest_data['total_cases']) * 100

print("COVID-19 Summary Statistics:")
print(summary_stats)

# Correlation analysis
correlation = filtered_data.groupby('location')[['new_cases', 'new_deaths']].corr().iloc[::2, 1].reset_index()
correlation.columns = ['location', 'correlation_cases_deaths']
print("\nCorrelation between new cases and new deaths:")
print(correlation)

# Vaccination progress
vax_data = filtered_data.groupby('location')[['people_fully_vaccinated_per_hundred']].max().reset_index()
print("\nVaccination Progress (% fully vaccinated):")
print(vax_data)
