# Roadmap to Master Required Skills for the Data Scientist Role

## 1. Python Programming Fundamentals & Pandas
Key Resources:

Official Python Tutorial – docs.python.org/3/tutorial

Real Python “Python Basics” – realpython.com/python-basics

Pandas Documentation – pandas.pydata.org/docs

DataCamp Pandas Tutorial – datacamp.com/community/tutorials/pandas-tutorial-dataframe-python

### Examples:

#### Basic data types and operations

```python
# integers, floats, strings, lists, dicts
x = 42
y = 3.14
name = "Data Scientist"
items = [1, 2, 3]
config = {"learning_rate": 0.01, "batch_size": 32}
print(type(x), type(items), config["learning_rate"])
List comprehensions and generator expressions

squares = [i**2 for i in range(10)]
even_gen = (i for i in range(10) if i % 2 == 0)
print(squares, list(even_gen))
```

#### Functions, modules, and error handling

```python
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

if __name__ == "__main__":
    print(safe_divide(10, 2), safe_divide(5, 0))
```

#### Creating and inspecting a DataFrame

```python
import pandas as pd

data = {
    'age': [25, 32, 47],
    'salary': [50000, 64000, 120000]
}
df = pd.DataFrame(data)
print(df.head(), df.describe())
```

#### Grouping and aggregating data

```python
import pandas as pd

df = pd.DataFrame({
    'department': ['IT', 'HR', 'IT', 'HR'],
    'salary': [70000, 50000, 80000, 52000]
})
grouped = df.groupby('department')['salary'].agg(['mean', 'max'])
print(grouped)
```


## 2. Machine Learning with scikit-learn
Key Resources:

scikit-learn User Guide – scikit-learn.org/stable/user_guide

DataCamp scikit-learn Tutorial – datacamp.com/community/tutorials/machine-learning-python

### Examples:

Train/Test Split & Standardization

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
print("Accuracy:", clf.score(scaler.transform(X_test), y_test))
Gradient Boosting Machine (GBM)

from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm.fit(X_train_scaled, y_train)
print("GBM Accuracy:", gbm.score(scaler.transform(X_test), y_test))
Voting Ensemble of Multiple Models

from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('rf', clf), ('gbm', gbm)],
    voting='soft'
)
voting_clf.fit(X_train_scaled, y_train)
print("Ensemble Accuracy:", voting_clf.score(scaler.transform(X_test), y_test))
Grid Search Cross-Validation

from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train_scaled, y_train)
print("Best params:", grid.best_params_, "Best score:", grid.best_score_)
## 3. Deep Learning with TensorFlow
Key Resources:

TensorFlow Tutorials – tensorflow.org/tutorials

TensorFlow Guide – tensorflow.org/guide

### Examples:

Simple DNN on MNIST

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
Convolutional Neural Network (CNN)

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28,28,1), input_shape=(28,28)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
model.fit(x_train, y_train, epochs=3)
Simple RNN for Sequence Data

import numpy as np
x = np.random.randn(1000, 10, 1)  # 1000 samples, 10 timesteps
y = np.random.randint(0, 2, size=(1000, 1))
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(16, input_shape=(10,1)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile('adam', 'binary_crossentropy', ['accuracy'])
model.fit(x, y, epochs=3)
Early Stopping Callback

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stop])
Transfer Learning with Pretrained Model

base = tf.keras.applications.MobileNetV2(input_shape=(128,128,3),
                                         include_top=False, pooling='avg')
base.trainable = False
inputs = tf.keras.Input(shape=(128,128,3))
x = base(inputs, training=False)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
tl_model = tf.keras.Model(inputs, outputs)
tl_model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
## 4. Deep Learning with PyTorch
Key Resources:

PyTorch Tutorials – pytorch.org/tutorials

Deep Learning with PyTorch: A 60 Minute Blitz – pytorch.org/tutorials/beginner/blitz

### Examples:

Tensor Operations

import torch
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.randn(2,2)
print(a + b, torch.matmul(a, b))
Simple Feedforward Network

import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
model = Net()
Training Loop

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(2):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
Convolutional Network

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16*13*13, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 16*13*13)
        return self.fc(x)
cnn = CNN()
Saving & Loading Models

torch.save(model.state_dict(), 'model.pth')
model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()
## 5. SQL & Database Performance Tuning
Key Resources:

PostgreSQL Tutorial – postgresql.org/docs/current/tutorial.html

Mode SQL Tutorial – mode.com/sql-tutorial

### Examples:

Basic SELECT with JOIN

import sqlite3
conn = sqlite3.connect('data.db')
query = """
SELECT a.id, b.value
FROM table_a a
JOIN table_b b ON a.id = b.a_id
"""
df = pd.read_sql_query(query, conn)
Window Function (ROW_NUMBER)

query = """
SELECT
  id,
  value,
  ROW_NUMBER() OVER (PARTITION BY category ORDER BY value DESC) as rn
FROM data_table
"""
df = pd.read_sql_query(query, conn)
Subquery in WHERE Clause

query = """
SELECT *
FROM sales
WHERE amount > (SELECT AVG(amount) FROM sales)
"""
df = pd.read_sql_query(query, conn)
Creating Index via psycopg2

import psycopg2
conn = psycopg2.connect("dbname=test user=postgres")
cur = conn.cursor()
cur.execute("CREATE INDEX idx_sales_date ON sales(date);")
conn.commit()
Explaining Query Plan

query = "EXPLAIN ANALYZE SELECT * FROM sales WHERE amount > 1000;"
plan = pd.read_sql_query(query, conn)
print(plan)
## 6. Data Engineering: Pipelines & Data Lakes/Warehouses
Key Resources:

Apache Airflow Docs – airflow.apache.org/docs

AWS Data Lake Guide – docs.aws.amazon.com/

Snowflake Documentation – docs.snowflake.com

### Examples:

Processing Large CSV in Chunks

for chunk in pd.read_csv('large.csv', chunksize=100000):
    process(chunk)
Airflow DAG Skeleton

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def task_func(): pass

with DAG('my_dag', start_date=datetime(2025,4,1), schedule_interval='@daily') as dag:
    task = PythonOperator(task_id='task', python_callable=task_func)
Reading/Writing Parquet

df = pd.read_parquet('data.parquet')
df.to_parquet('out.parquet', index=False)
Uploading to AWS S3

import boto3
s3 = boto3.client('s3')
s3.upload_file('out.parquet', 'my-bucket', 'data/out.parquet')
Writing to a Data Warehouse via SQLAlchemy

from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@host:5432/db')
df.to_sql('table_name', engine, if_exists='replace', index=False)
## 7. Model Deployment with Python Web Frameworks
Key Resources:

Flask Documentation – flask.palletsprojects.com

FastAPI Tutorial – fastapi.tiangolo.com/tutorial

Django REST Framework – www.django-rest-framework.org

### Examples:

Simple Flask API

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = model.predict([data['features']])
    return jsonify({'prediction': result[0]})

if __name__ == '__main__':
    app.run(debug=True)

FastAPI Endpoint with Pydantic

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    features: list[float]

app = FastAPI()

@app.post("/predict")
def predict(item: Item):
    pred = model.predict([item.features])
    return {"prediction": int(pred[0])}
```

Django View for Prediction

```python
# views.py
from django.http import JsonResponse
import json

def predict(request):
    body = json.loads(request.body)
    pred = model.predict([body['features']])
    return JsonResponse({'prediction': pred[0]})
```

Dockerfile for Flask App
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

Client Request to API

import requests

resp = requests.post(
    "http://localhost:5000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(resp.json())
## 8. Kaggle Competitions & Real-world Projects
Key Resources:

Kaggle Learn – kaggle.com/learn

Kaggle API Docs – github.com/Kaggle/kaggle-api

### Examples:

Downloading a Dataset via Kaggle API

kaggle competitions download -c titanic
Exploratory Data Analysis (EDA)

import pandas as pd
df = pd.read_csv('train.csv')
print(df.info(), df.describe(), df['Survived'].value_counts())
Baseline Model & Submission

from sklearn.tree import DecisionTreeClassifier

X = df.drop(['Survived','PassengerId'], axis=1).select_dtypes(include='number')
y = df['Survived']
clf = DecisionTreeClassifier().fit(X, y)
test = pd.read_csv('test.csv')
preds = clf.predict(test[X.columns])
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds})
submission.to_csv('submission.csv', index=False)
Feature Engineering Example

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
Cross-Validation for Robust Evaluation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print("CV scores:", scores, "Mean:", scores.mean())
