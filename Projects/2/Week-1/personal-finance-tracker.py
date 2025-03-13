import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
import random

# Generate sample spending data
def generate_sample_data(months=6):
    np.random.seed(42)
    
    # Categories
    categories = [
        'Housing', 'Groceries', 'Dining Out', 'Transportation', 
        'Utilities', 'Entertainment', 'Shopping', 'Healthcare',
        'Education', 'Savings', 'Miscellaneous'
    ]
    
    # Create date range (last few months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*months)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Transaction data
    transactions = []
    
    # Income (monthly)
    monthly_income = 5000
    for month in range(months):
        income_date = start_date + timedelta(days=month*30)
        transactions.append({
            'Date': income_date,
            'Category': 'Income',
            'Description': 'Salary',
            'Amount': monthly_income,
            'Type': 'Income'
        })
    
    # Regular expenses
    for date in date_range:
        # Rent/Mortgage (monthly)
        if date.day == 1:
            transactions.append({
                'Date': date,
                'Category': 'Housing',
                'Description': 'Rent/Mortgage',
                'Amount': -1500,
                'Type': 'Expense'
            })
        
        # Groceries (2-3 times per week)
        if date.weekday() in [0, 3] and random.random() < 0.8:
            transactions.append({
                'Date': date,
                'Category': 'Groceries',
                'Description': f'Groceries - {random.choice(["Supermarket", "Local Store", "Farmers Market"])}',
                'Amount': -random.randint(50, 120),
                'Type': 'Expense'
            })
        
        # Dining out (1-2 times per week)
        if date.weekday() in [4, 5] and random.random() < 0.6:
            transactions.append({
                'Date': date,
                'Category': 'Dining Out',
                'Description': f'Restaurant - {random.choice(["Dinner", "Lunch", "Brunch"])}',
                'Amount': -random.randint(25, 80),
                'Type': 'Expense'
            })
        
        # Transportation (weekdays)
        if date.weekday() < 5:
            transactions.append({
                'Date': date,
                'Category': 'Transportation',
                'Description': random.choice(['Gas', 'Public Transit', 'Rideshare']),
                'Amount': -random.randint(5, 30),
                'Type': 'Expense'
            })
        
        # Utilities (monthly)
        if date.day == 15:
            for utility in ['Electricity', 'Water', 'Internet', 'Phone']:
                transactions.append({
                    'Date': date,
                    'Category': 'Utilities',
                    'Description': utility,
                    'Amount': -random.randint(40, 100),
                    'Type': 'Expense'
                })
        
        # Entertainment (weekends)
        if date.weekday() >= 5 and random.random() < 0.7:
            transactions.append({
                'Date': date,
                'Category': 'Entertainment',
                'Description': random.choice(['Movies', 'Streaming', 'Concerts', 'Sports']),
                'Amount': -random.randint(15, 60),
                'Type': 'Expense'
            })
        
        # Shopping (random)
        if random.random() < 0.2:
            transactions.append({
                'Date': date,
                'Category': 'Shopping',
                'Description': random.choice(['Clothing', 'Electronics', 'Home Goods', 'Books']),
                'Amount': -random.randint(30, 200),
                'Type': 'Expense'
            })
        
        # Healthcare (monthly)
        if date.day == 20 and random.random() < 0.3:
            transactions.append({
                'Date': date,
                'Category': 'Healthcare',
                'Description': random.choice(['Doctor Visit', 'Pharmacy', 'Insurance']),
                'Amount': -random.randint(20, 150),
                'Type': 'Expense'
            })
        
        # Add some randomness
        if random.random() < 0.1:
            category = random.choice(categories)
            transactions.append({
                'Date': date,
                'Category': category,
                'Description': f'Miscellaneous {category}',
                'Amount': -random.randint(10, 100),
                'Type': 'Expense'
            })
    
    # Create DataFrame
    transactions_df = pd.DataFrame(transactions)
    transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])
    
    return transactions_df

# Main finance tracker class
class PersonalFinanceTracker:
    def __init__(self, data=None):
        if data is None:
            self.transactions = generate_sample_data()
        else:
            self.transactions = data
            
        # Ensure date is datetime
        self.transactions['Date'] = pd.to_datetime(self.transactions['Date'])
        
        # Sort by date
        self.transactions = self.transactions.sort_values('Date')

    def add_transaction(self, date, category, description, amount, transaction_type=None):
        """Add a new transaction to the tracker."""
        if transaction_type is None:
            transaction_type = 'Income' if amount > 0 else 'Expense'
            
        new_transaction = {
            'Date': pd.to_datetime(date),
            'Category': category,
            'Description': description,
            'Amount': amount,
            'Type': transaction_type
        }
        
        self.transactions = pd.concat([self.transactions, pd.DataFrame([new_transaction])])
        self.transactions = self.transactions.sort_values('Date')
        print(f"Added new {transaction_type}: {description}")

    def get_balance(self):
        """Calculate current balance."""
        return self.transactions['Amount'].sum()

    def get_monthly_summary(self):
        """Get summary of income and expenses by month."""
        # Create month column
        self.transactions['Month'] = self.transactions['Date'].dt.to_period('M')
        
        # Group by month and type
        monthly = self.transactions.groupby(['Month', 'Type'])['Amount'].sum().unstack().fillna(0)
        
        # Calculate net (income - expenses)
        if 'Expense' in monthly.columns and 'Income' in monthly.columns:
            monthly['Net'] = monthly['Income'] + monthly['Expense']  # Expense is negative
        
        return monthly

    def plot_monthly_summary(self):
        """Plot monthly income, expenses, and savings."""
        monthly = self.get_monthly_summary()
        
        plt.figure(figsize=(12, 6))
        
        # Convert period index to datetime for better plotting
        monthly.index = monthly.index.to_timestamp()
        
        if 'Income' in monthly.columns:
            plt.bar(monthly.index, monthly['Income'], color='green', alpha=0.7, label='Income')
        
        if 'Expense' in monthly.columns:
            # Convert expenses to positive for visualization
            plt.bar(monthly.index, -monthly['Expense'], color='red', alpha=0.7, label='Expenses')
        
        if 'Net' in monthly.columns:
            plt.plot(monthly.index, monthly['Net'], 'bo-', label='Net')
        
        plt.title('Monthly Income and Expenses')
        plt.xlabel('Month')
        plt.ylabel('Amount ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_expense_breakdown(self, period='all'):
        """Plot breakdown of expenses by category."""
        # Filter for expenses only
        expenses = self.transactions[self.transactions['Type'] == 'Expense'].copy()
        
        if period != 'all':
            # Filter for the last n months
            n_months = int(period)
            start_date = datetime.now() - timedelta(days=30*n_months)
            expenses = expenses[expenses['Date'] >= start_date]
        
        # Group by category
        category_expenses = expenses.groupby('Category')['Amount'].sum()
        
        # Convert to positive for visualization
        category_expenses = -category_expenses
        
        # Sort by amount
        category_expenses = category_expenses.sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(category_expenses.index, category_expenses.values, color=sns.color_palette('viridis', len(category_expenses)))
        
        # Add percentage labels
        total = category_expenses.sum()
        for bar in bars:
            height = bar.get_height()
            percentage = height / total * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.title('Expense Breakdown by Category')
        plt.xlabel('Category')
        plt.ylabel('Amount ($)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return category_expenses

    def plot_spending_trends(self):
        """Plot spending trends over time by category."""
        # Filter for expenses and create month column
        expenses = self.transactions[self.transactions['Type'] == 'Expense'].copy()
        expenses['Month'] = expenses['Date'].dt.to_period('M')
        
        # Group by month and category
        category_monthly = expenses.groupby(['Month', 'Category'])['Amount'].sum().unstack().fillna(0)
        
        # Convert to positive for visualization
        category_monthly = -category_monthly
        
        # Convert period index to datetime for better plotting
        category_monthly.index = category_monthly.index.to_timestamp()
        
        plt.figure(figsize=(12, 6))
        
        # Plot top 5 categories
        top_categories = category_monthly.sum().sort_values(ascending=False).head(5).index
        
        for category in top_categories:
            plt.plot(category_monthly.index, category_monthly[category], 'o-', label=category)
        
        plt.title('Monthly Spending Trends by Category')
        plt.xlabel('Month')
        plt.ylabel('Amount ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_savings_rate(self):
        """Calculate savings rate (savings as % of income)."""
        income = self.transactions[self.transactions['Type'] == 'Income']['Amount'].sum()
        expenses = -self.transactions[self.transactions['Type'] == 'Expense']['Amount'].sum()
        
        if income > 0:
            savings = income - expenses
            savings_rate = (savings / income) * 100
            return savings_rate
        else:
            return 0

    def display_dashboard(self):
        """Display a comprehensive financial dashboard."""
        print("===== PERSONAL FINANCE DASHBOARD =====")
        
        # Current Balance
        balance = self.get_balance()
        print(f"Current Balance: ${balance:.2f}")
        
        # Monthly Summary
        monthly = self.get_monthly_summary()
        print("\nMonthly Summary:")
        print(monthly)
        
        # Savings Rate
        savings_rate = self.get_savings_rate()
        print(f"\nOverall Savings Rate: {savings_rate:.1f}%")
        
        # Recent Transactions
        print("\nRecent Transactions:")
        recent = self.transactions.tail(5)[['Date', 'Category', 'Description', 'Amount']]
        recent['Date'] = recent['Date'].dt.strftime('%Y-%m-%d')
        print(recent)
        
        # Visualizations
        print("\nGenerating visualizations...")
        
        # Monthly summary plot
        self.plot_monthly_summary()
        
        # Expense breakdown
        category_expenses = self.plot_expense_breakdown(period='3')
        
        # Spending trends
        self.plot_spending_trends()
        
        # Recommendations
        print("\nFinancial Recommendations:")
        top_expense = category_expenses.index[0]
        print(f"1. Your highest expense category is '{top_expense}' at ${category_expenses.iloc[0]:.2f}.")
        print(f"   Consider ways to reduce spending in this area.")
        
        if savings_rate