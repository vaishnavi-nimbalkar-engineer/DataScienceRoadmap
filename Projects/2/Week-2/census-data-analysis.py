import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate synthetic census data
np.random.seed(42)
n_records = 1000

# Define regions and demographics
regions = ['Northeast', 'Midwest', 'South', 'West']
age_ranges = ['0-18', '19-35', '36-50', '51-65', '66+']
education_levels = ['Less than High School', 'High School', 'Some College', 'Bachelor\'s', 'Graduate']
income_brackets = ['<$25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '$100k+']

# Generate random data
region = np.random.choice(regions, size=n_records, p=[0.2, 0.25, 0.35, 0.2])
age_range = np.random.choice(age_ranges, size=n_records, p=[0.2, 0.25, 0.25, 0.2, 0.1])
gender = np.random.choice(['Male', 'Female', 'Other'], size=n_records, p=[0.49, 0.5, 0.01])
education = np.random.choice(education_levels, size=n_records, p=[0.1, 0.3, 0.2, 0.3, 0.1])
marital_status = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                 size=n_records, p=[0.4, 0.4, 0.15, 0.05])

# Generate incomes based on education level
def generate_income(education_level):
    base = {
        'Less than High School': 20000,
        'High School': 35000,
        'Some College': 45000,
        'Bachelor\'s': 60000,
        'Graduate': 85000
    }
    
    # Add some randomness
    mean = base[education_level]
    std = mean * 0.3  # 30% standard deviation
    
    return max(10000, int(np.random.normal(mean, std)))

incomes = np.array([generate_income(e) for e in education])

# Generate income bracket based on income
def get_income_bracket(income):
    if income < 25000:
        return '<$25k'
    elif income < 50000:
        return '$25k-$50k'
    elif income < 75000:
        return '$50k-$75k'
    elif income < 100000:
        return '$75k-$100k'
    else:
        return '$100k+'

income_bracket = np.array([get_income_bracket(i) for i in incomes])

# Generate household size based on marital status
def generate_household_size(marital_status):
    if marital_status == 'Single':
        return max(1, int(np.random.normal(1.5, 0.8)))
    elif marital_status == 'Married':
        return max(2, int(np.random.normal(3.5, 1.2)))
    elif marital_status == 'Divorced':
        return max(1, int(np.random.normal(2.2, 1.0)))
    else:  # Widowed
        return max(1, int(np.random.normal(1.8, 0.9)))

household_size = np.array([generate_household_size(m) for m in marital_status])

# Create DataFrame
census_data = pd.DataFrame({
    'Region': region,
    'Age_Range': age_range,
    'Gender': gender,
    'Education': education,
    'Marital_Status': marital_status,
    'Income': incomes,
    'Income_Bracket': income_bracket,
    'Household_Size': household_size
})

# Class for census data analysis
class CensusAnalyzer:
    def __init__(self, data):
        self.data = data
        self.numeric_columns = ['Income', 'Household_Size']
        self.categorical_columns = ['Region', 'Age_Range', 'Gender', 'Education', 
                                   'Marital_Status', 'Income_Bracket']
    
    def summary_statistics(self):
        """Calculate and display summary statistics for numeric variables."""
        print("Summary Statistics for Numeric Variables:")
        print("-" * 60)
        
        for column in self.numeric_columns:
            data = self.data[column]
            
            # Central tendency
            mean = data.mean()
            median = data.median()
            mode = data.mode()[0]
            
            # Dispersion
            std_dev = data.std()
            variance = data.var()
            min_val = data.min()
            max_val = data.max()
            range_val = max_val - min_val
            
            # Percentiles
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            # Print results
            print(f"Statistics for {column}:")
            print(f"  Mean: {mean:.2f}")
            print(f"  Median: {median:.2f}")
            print(f"  Mode: {mode}")
            print(f"  Standard Deviation: {std_dev:.2f}")
            print(f"  Variance: {variance:.2f}")
            print(f"  Range: {range_val} (Min: {min_val}, Max: {max_val})")
            print(f"  Interquartile Range: {iqr:.2f} (Q1: {q1:.2f}, Q3: {q3:.2f})")
            
            # Check for skewness
            skewness = stats.skew(data)
            if abs(skewness) < 0.5:
                skew_desc = "approximately symmetric"
            elif skewness < 0:
                skew_desc = "negatively skewed"
            else:
                skew_desc = "positively skewed"
            
            print(f"  Skewness: {skewness:.2f} ({skew_desc})")
            print("-" * 60)
            
        return
    
    def categorical_statistics(self):
        """Analyze categorical variables."""
        print("Categorical Variable Analysis:")
        print("-" * 60)
        
        for column in self.categorical_columns:
            counts = self.data[column].value_counts()
            proportions = self.data[column].value_counts(normalize=True) * 100
            
            print(f"Distribution of {column}:")
            summary = pd.DataFrame({
                'Count': counts,
                'Percentage': proportions
            })
            print(summary)
            print("-" * 60)
            
        return
    
    def cross_tabulation(self, var1, var2):
        """Create a cross-tabulation of two categorical variables."""
        cross_tab = pd.crosstab(
            self.data[var1], 
            self.data[var2], 
            normalize='index'
        ) * 100
        
        print(f"Cross-tabulation of {var1} by {var2} (row percentages):")
        print(cross_tab)
        
        return cross_tab
    
    def plot_histogram(self, column, bins=10):
        """Plot histogram for a numeric variable."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], bins=bins, kde=True)
        
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return
    
    def plot_boxplot(self, numeric_var, group_var=None):
        """Plot boxplot for a numeric variable, optionally grouped by a categorical variable."""
        plt.figure(figsize=(12, 6))
        
        if group_var:
            sns.boxplot(x=group_var, y=numeric_var, data=self.data)
            plt.title(f'Distribution of {numeric_var} by {group_var}')
            plt.xticks(rotation=45)
        else:
            sns.boxplot(y=numeric_var, data=self.data)
            plt.title(f'Distribution of {numeric_var}')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return
    
    def plot_bar_chart(self, column):
        """Plot bar chart for a categorical variable."""
        counts = self.data[column].value_counts().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values)
        
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return
    
    def plot_heatmap(self, var1, var2):
        """Plot heatmap for two categorical variables."""
        cross_tab = pd.crosstab(self.data[var1], self.data[var2])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='viridis')
        
        plt.title(f'Heatmap of {var1} vs {var2}')
        plt.tight_layout()
        plt.show()
        
        return
    
    def income_by_education(self):
        """Analyze income by education level."""
        income_by_edu = self.data.groupby('Education')['Income'].agg(['mean', 'median', 'std'])
        income_by_edu = income_by_edu.sort_values('mean')
        
        print("Income Statistics by Education Level:")
        print(income_by_edu)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Education', y='Income', data=self.data, estimator=np.mean, 
                   order=income_by_edu.index)
        
        plt.title('Average Income by Education Level')
        plt.xlabel('Education Level')
        plt.ylabel('Average Income ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return income_by_edu
    
    def regional_demographics(self):
        """Analyze demographic distributions by region."""
        # Age distribution by region
        age_by_region = pd.crosstab(self.data['Region'], self.data['Age_Range'], normalize='index') * 100
        
        plt.figure(figsize=(12, 6))
        age_by_region.plot(kind='bar', stacked=True, colormap='viridis')
        
        plt.title('Age Distribution by Region')
        plt.xlabel('Region')
        plt.ylabel('Percentage')
        plt.legend(title='Age Range')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Income brackets by region
        income_by_region = pd.crosstab(self.data['Region'], self.data['Income_Bracket'], normalize='index') * 100
        
        plt.figure(figsize=(12, 6))
        income_by_region.plot(kind='bar', stacked=True, colormap='plasma')
        
        plt.title('Income Distribution by Region')
        plt.xlabel('Region')
        plt.ylabel('Percentage')
        plt.legend(title='Income Bracket')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return age_by_region, income_by_region
    
    def run_analysis(self):
        """Run a comprehensive analysis on the census data."""
        print("====== CENSUS DATA ANALYSIS ======")
        print(f"Total Records: {len(self.data)}")
        print("\n")
        
        # Summary statistics
        self.summary_statistics()
        
        # Categorical statistics
        self.categorical_statistics()
        
        # Income distribution
        self.plot_histogram('Income', bins=20)
        
        # Household size distribution
        self.plot_histogram('Household_Size', bins=10)
        
        # Income by education
        self.income_by_education()
        
        # Income by region
        self.plot_boxplot('Income', 'Region')
        
        # Regional demographics
        self.regional_demographics()
        
        # Cross-tabulation of education and income bracket
        self.cross_tabulation('Education', 'Income_Bracket')
        self.plot_heatmap('Education', 'Income_Bracket')
        
        # Marital status and household size
        self.plot_boxplot('Household_Size', 'Marital_Status')
        
        # Gender distribution
        self.plot_bar_chart('Gender')
        
        # Key findings
        print("\nKEY FINDINGS:")
        
        # Education and income relationship
        edu_income = self.data.groupby('Education')['Income'].mean().sort_values()
        lowest_edu = edu_income.index[0]
        highest_edu = edu_income.index[-1]
        income_diff = edu_income[highest_edu] - edu_income[lowest_edu]
        income_ratio = edu_income[highest_edu] / edu_income[lowest_edu]
        
        print(f"1. Education significantly impacts income. On average, individuals with {highest_edu} education")
        print(f"   earn ${income_diff:.2f} more than those with {lowest_edu} education")
        print(f"   (a {income_ratio:.2f}x difference).")
        
        # Regional differences
        region_income = self.data.groupby('Region')['Income'].mean().sort_values()
        lowest_region = region_income.index[0]
        highest_region = region_income.index[-1]
        
        print(f"2. Regional income disparities exist. The {highest_region} region has the highest average income,")
        print(f"   while the {lowest_region} region has the lowest.")
        
        # Household size and marital status
        household_by_status = self.data.groupby('Marital_Status')['Household_Size'].mean().sort_values()
        
        print("3. Household size varies by marital status:")
        for status, size in household_by_status.items():
            print(f"   - {status}: {size:.2f} persons on average")
        
        return

# Run the analysis
if __name__ == "__main__":
    analyzer = CensusAnalyzer(census_data)
    analyzer.run_analysis()
