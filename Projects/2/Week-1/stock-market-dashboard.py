import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import seaborn as sns

# Set the style for plots
plt.style.use('fivethirtyeight')

# Define the stocks to track
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
stock_names = {
    'AAPL': 'Apple', 
    'MSFT': 'Microsoft', 
    'GOOGL': 'Google', 
    'AMZN': 'Amazon', 
    'META': 'Meta'
}

# Define the time period
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Last year

# Download stock data
stock_data = {}
for stock in stocks:
    stock_data[stock] = yf.download(stock, start=start_date, end=end_date)

# Create a function to plot stock price trends
def plot_stock_prices():
    plt.figure(figsize=(14, 7))
    
    for stock in stocks:
        # Normalize to percentage change from first day
        first_price = stock_data[stock]['Adj Close'].iloc[0]
        normalized = (stock_data[stock]['Adj Close'] / first_price - 1) * 100
        plt.plot(normalized, label=stock_names[stock])
    
    plt.title('Stock Performance (% Change)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('% Change', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Function to calculate and display key metrics
def display_metrics():
    metrics = []
    
    for stock in stocks:
        data = stock_data[stock]
        
        # Calculate metrics
        current_price = data['Adj Close'].iloc[-1]
        previous_price = data['Adj Close'].iloc[-2]
        start_price = data['Adj Close'].iloc[0]
        
        daily_change = (current_price - previous_price) / previous_price * 100
        yearly_change = (current_price - start_price) / start_price * 100
        
        # Calculate volatility (standard deviation of daily returns)
        daily_returns = data['Adj Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized, in percentage
        
        # 50-day moving average
        ma_50 = data['Adj Close'].rolling(window=50).mean().iloc[-1]
        ma_signal = "Above 50-day MA" if current_price > ma_50 else "Below 50-day MA"
        
        # Store metrics
        metrics.append({
            'Stock': stock_names[stock],
            'Current Price': f"${current_price:.2f}",
            'Daily Change': f"{daily_change:.2f}%",
            'Yearly Change': f"{yearly_change:.2f}%",
            'Volatility': f"{volatility:.2f}%",
            'MA Signal': ma_signal
        })
    
    # Create a DataFrame for display
    metrics_df = pd.DataFrame(metrics)
    
    print("Stock Market Performance Metrics:")
    print("-" * 100)
    print(metrics_df.to_string(index=False))
    print("-" * 100)
    
    return metrics_df

# Function to plot trading volume
def plot_volume():
    plt.figure(figsize=(14, 7))
    
    for stock in stocks:
        # Get monthly volume
        monthly_volume = stock_data[stock]['Volume'].resample('M').sum()
        plt.bar(monthly_volume.index, monthly_volume, label=stock_names[stock], alpha=0.7)
    
    plt.title('Monthly Trading Volume', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volume', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Function to visualize correlation between stocks
def plot_correlation():
    # Combine all stock prices
    all_prices = pd.DataFrame()
    
    for stock in stocks:
        all_prices[stock] = stock_data[stock]['Adj Close']
    
    # Calculate correlation
    correlation = all_prices.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Stock Price Correlation', fontsize=16)
    plt.tight_layout()
    plt.show()

# Create the dashboard
def run_dashboard():
    print("===== Stock Market Performance Dashboard =====")
    print(f"Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("\n")
    
    # Display metrics table
    metrics_df = display_metrics()
    
    # Plot stock price trends
    plot_stock_prices()
    
    # Plot trading volume
    plot_volume()
    
    # Plot correlation
    plot_correlation()
    
    print("\nDashboard Summary:")
    # Identify best and worst performers
    best_stock = metrics_df.loc[metrics_df['Yearly Change'].str.rstrip('%').astype(float).idxmax()]
    worst_stock = metrics_df.loc[metrics_df['Yearly Change'].str.rstrip('%').astype(float).idxmin()]
    
    print(f"Best Performer: {best_stock['Stock']} with {best_stock['Yearly Change']} yearly change")
    print(f"Worst Performer: {worst_stock['Stock']} with {worst_stock['Yearly Change']} yearly change")

# Run the dashboard
if __name__ == "__main__":
    run_dashboard()
