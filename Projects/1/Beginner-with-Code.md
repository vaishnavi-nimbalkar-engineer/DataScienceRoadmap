### **Month 1: Foundations**

#### Week 1: Global Happiness Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('world_happiness.csv')

# Basic stats
print(df.describe())

# Correlation matrix
print(df.corr())

# Plot GDP vs Happiness Score
plt.scatter(df['GDP_per_capita'], df['Happiness_Score'])
plt.title('GDP vs Happiness')
plt.xlabel('GDP per Capita')
plt.ylabel('Happiness Score')
plt.show()
```


#### Week 2: Student Performance Stats

```python
import pandas as pd

data = {
    'Math': [88, 72, 95, 61, 84],
    'Science': [92, 78, 86, 68, 77]
}

df = pd.DataFrame(data)
print(f"Mean Math: {df['Math'].mean()}")
print(f"Median Science: {df['Science'].median()}")
print(f"Standard Deviation Math: {df['Math'].std()}")
```


#### Week 3: Dice Probability Simulation

```python
import numpy as np

# Simulate 10,000 dice rolls
rolls = np.random.randint(1, 7, 10000)

# Calculate probabilities
for i in range(1, 7):
    prob = np.mean(rolls == i)
    print(f"P({i}): {prob:.2%}")
```


#### Week 4: Text Data Cleaner

```python
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

sample_tweet = "Wow! This is SO cool!!! #DataScience2025"
print(clean_text(sample_tweet))  # Output: wow this is so cool datascience2025
```

---

### **Month 2: Data Manipulation**

#### Week 5: E-commerce Sales Analysis

```python
import pandas as pd

sales = pd.read_csv('ecommerce_sales.csv')
sales['Order_Date'] = pd.to_datetime(sales['Order_Date'])
monthly_sales = sales.groupby(sales['Order_Date'].dt.month)['Revenue'].sum()
print(monthly_sales.plot(kind='bar'))
```


#### Week 6: NumPy Linear Regression

```python
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Linear regression using NumPy
coefficients = np.polyfit(X, Y, 1)
print(f"Slope: {coefficients[0]:.2f}, Intercept: {coefficients[1]:.2f}")
```


#### Week 7: Data Cleaning Pipeline

```python
def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mean())  # Fill numeric NaNs with mean
    return df

# Usage:
raw_df = pd.read_csv('dirty_data.csv')
clean_df = clean_data(raw_df)
```


#### Week 8: CO2 Emissions Visualization

```python
import seaborn as sns

df = sns.load_dataset('co2')
sns.lineplot(x='year', y='co2', data=df)
plt.title('Global CO2 Emissions Over Time')
plt.show()
```

---

### **Month 3: Advanced Topics**

#### Week 9: Titanic EDA

```python
import seaborn as sns

titanic = sns.load_dataset('titanic')
sns.countplot(x='class', hue='survived', data=titanic)
plt.title('Survival by Class')
plt.show()
```


#### Week 10: Power BI Alternative (Python Dashboard)

```python
from dash import Dash, html, dcc
import plotly.express as px

app = Dash(__name__)
df = px.data.gapminder()

app.layout = html.Div([
    dcc.Graph(figure=px.scatter(df, x='gdpPercap', y='lifeExp', size='pop'))
])

if __name__ == '__main__':
    app.run_server(debug=True)
```


#### Week 11: SQL Analysis in Python

```python
import sqlite3

conn = sqlite3.connect('sales.db')
query = '''
    SELECT region, SUM(revenue) 
    FROM sales 
    GROUP BY region
'''
result = pd.read_sql_query(query, conn)
print(result)
```


#### Week 12: Airbnb Analysis Template

```python
airbnb = pd.read_csv('airbnb_listings.csv')

# Clean price column
airbnb['price'] = airbnb['price'].str.replace('$', '').astype(float)

# Top neighborhoods by average price
print(airbnb.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(10))
```

---

## Sentiment Analysis on Movie Reviews

Dataset: IMDb Movie Reviews
Project: Build a sentiment analysis model to classify movie reviews as positive or negative.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv('imdb_reviews.csv')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2)

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train and evaluate the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print(f"Accuracy: {model.score(X_test_vec, y_test):.2f}")
```


## Time Series Analysis on Stock Prices

Dataset: Yahoo Finance
Project: Analyze and forecast stock prices using ARIMA models.

```python
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Download stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2025-03-13')

# Fit ARIMA model
model = ARIMA(stock_data['Close'], order=(1,1,1))
results = model.fit()

# Forecast
forecast = results.forecast(steps=30)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(stock_data.index, stock_data['Close'], label='Historical')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.title('AAPL Stock Price Forecast')
plt.show()
```


## Image Classification with Convolutional Neural Networks

Dataset: CIFAR-10
Project: Build a CNN to classify images into 10 categories.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc:.2f}')
```


## Recommendation System

Dataset: MovieLens
Project: Build a collaborative filtering recommendation system for movies.

```python
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load the MovieLens dataset
ratings = pd.read_csv('ratings.csv')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data
trainset, testset = train_test_split(data, test_size=0.25)

# Train the SVD model
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Evaluate the model
from surprise import accuracy
accuracy.rmse(predictions)
```

These examples cover a range of data science tasks and use different datasets. Here are some additional dataset recommendations:

1. Kaggle's "Spotify Dataset 1921-2020, 160k+ Tracks" for music analysis
2. UCI Machine Learning Repository's "Adult" dataset for income prediction
3. World Bank Open Data for global development analysis
4. NASA's Earth Data for climate and environmental projects
5. Twitter API for social media sentiment analysis
6. Yelp Dataset for business analytics and NLP tasks

Remember to check the licensing and usage terms for each dataset before using them in your projects[^1][^4][^6].

<div style="text-align: center">⁂</div>

[^1]: https://www.stratascratch.com/blog/19-data-science-project-ideas-for-beginners/

[^2]: https://sherbold.github.io/intro-to-data-science/exercises/Solution_Data_Exploration.html

[^3]: https://github.com/nikopetr/Data-Science-Probability-and-Statistics-Projects

[^4]: https://www.projectpro.io/article/python-projects-for-data-science/462

[^5]: https://github.com/schlende/practical-pandas-projects

[^6]: https://www.projectpro.io/article/datasets-for-data-science-projects/953

[^7]: https://www.youtube.com/watch?v=krkS9u140tM

[^8]: https://365datascience.com/trending/public-datasets-machine-learning/

[^9]: https://www.dataquest.io/blog/free-datasets-for-projects/

[^10]: https://sigma.ai/open-datasets/

[^11]: https://careerfoundry.com/en/blog/data-analytics/where-to-find-free-datasets/

[^12]: https://365datascience.com/trending/free-dataset-resources/

[^13]: https://www.interviewquery.com/p/free-datasets

[^14]: https://www.projectpro.io/article/15-data-science-projects-for-beginners-with-source-code/343

[^15]: https://www.investopedia.com/terms/d/descriptive_statistics.asp

[^16]: https://resources.experfy.com/ai-ml/useful-probability-distributions-with-applications-to-data-science-problems/

[^17]: https://www.projectpro.io/article/python-pandas-project-ideas/580

[^18]: https://www.youtube.com/watch?v=l6DOy4U4xIw

[^19]: https://sunscrapers.com/blog/data-cleaning-with-examples/

[^20]: https://365datascience.com/trending/data-visualization-project-ideas/

[^21]: https://www.tableau.com/learn/articles/free-public-data-sets

[^22]: https://www.kaggle.com/datasets?tags=16639-Python

[^23]: https://www.kaggle.com/general/260690

[^24]: https://www.reddit.com/r/datascience/comments/uyz6cz/list_of_open_source_data_sources/

[^25]: https://builtin.com/data-science/free-datasets

[^26]: https://www.youtube.com/watch?v=B0FnVlYUcI8

[^27]: https://www.dataquest.io/tutorial/data-cleaning-project-walk-through/

[^28]: https://www.guvi.in/blog/data-visualization-project-ideas/

