import pandas as pd     
# Create a DataFrame with missing values
df.fillna(0, inplace=True)  # Replace NaN with 0
# or
df.dropna(inplace=True)     # Drop rows with NaN
