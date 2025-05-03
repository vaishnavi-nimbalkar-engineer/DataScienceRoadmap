# Data Analytics Assessment Questions with Answers

## Mathematics

### What is the value of π to 2 decimal places?
<details>
<summary><strong>Show Answer</strong></summary>

3.14

</details>

### What is the derivative of sin(x)?
<details>
<summary><strong>Show Answer</strong></summary>

cos(x)

</details>

### Solve for x: 2x + 3 = 11
<details>
<summary><strong>Show Answer</strong></summary>

x = 4

</details>

### What is the integral of 1/x dx?
<details>
<summary><strong>Show Answer</strong></summary>

ln|x| + C

</details>

### What is matrix multiplication?
<details>
<summary><strong>Show Answer</strong></summary>

A process of multiplying rows of the first matrix with columns of the second

</details>

## Python (General)

### What is the output of len('Data')?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(len('Data'))  # Output: 4
```

4

</details>

### What is the difference between a list and a tuple?
<details>
<summary><strong>Show Answer</strong></summary>

```python
my_list = [1, 2, 3]    # Mutable
my_tuple = (1, 2, 3)   # Immutable
```

Lists are mutable, tuples are immutable.

</details>

### What is a lambda function?
<details>
<summary><strong>Show Answer</strong></summary>

```python
add = lambda x, y: x + y
print(add(2, 3))  # Output: 5
```

An anonymous function defined with the lambda keyword.

</details>

### What is a list comprehension?
<details>
<summary><strong>Show Answer</strong></summary>

```python
squares = [x**2 for x in range(5)]
print(squares)  # Output: [0, 1, 4, 9, 16]
```

A concise way to create lists using a single line of code.

</details>

### How do you handle exceptions in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
```

Using try and except blocks.

</details>

## NumPy

### How do you create a NumPy array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
import numpy as np
arr = np.array([1, 2, 3])
```

Using np.array()

</details>

### What function returns the shape of an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([[1, 2], [3, 4]])
print(arr.shape)  # Output: (2, 2)
```

array.shape

</details>

### How to generate a 2D array of zeros?
<details>
<summary><strong>Show Answer</strong></summary>

```python
zeros = np.zeros((2, 2))
print(zeros)
```

np.zeros((2,2))

</details>

### How do you compute the mean of an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 3])
print(np.mean(arr))  # Output: 2.0
```

np.mean(array)

</details>

### What is broadcasting in NumPy?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([1, 2, 3])
b = 2
print(a + b)  # Output: [3 4 5]
```

Automatic expansion of arrays to make their shapes compatible.

</details>

## Pandas

### How do you get the first 5 rows of a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6]})
print(df.head())
```

df.head()

</details>

### How do you filter rows based on a condition?
<details>
<summary><strong>Show Answer</strong></summary>

```python
filtered = df[df['A'] > 3]
print(filtered)
```

Using boolean indexing like df[df['col'] > 5]

</details>

### How to check for missing values?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.isnull())
```

df.isnull()

</details>

### How do you create a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
```

Using pd.DataFrame()

</details>

### What method is used to read a CSV file?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = pd.read_csv('file.csv')
```

pd.read_csv()

</details>

## Data Analysis Concepts

### What is normalization?
<details>
<summary><strong>Show Answer</strong></summary>

Rescaling values into a standard range.

</details>

### What is a KPI?
<details>
<summary><strong>Show Answer</strong></summary>

Key Performance Indicator.

</details>

### What is the difference between supervised and unsupervised learning?
<details>
<summary><strong>Show Answer</strong></summary>

Supervised uses labeled data, unsupervised does not.

</details>

### What is EDA?
<details>
<summary><strong>Show Answer</strong></summary>

Exploratory Data Analysis.

</details>

### What is data cleaning?
<details>
<summary><strong>Show Answer</strong></summary>

The process of fixing or removing incorrect, corrupted, or incomplete data.

</details>

## Statistics & Probability

### What is correlation?
<details>
<summary><strong>Show Answer</strong></summary>

A measure of the relationship between two variables.

</details>

### What is standard deviation?
<details>
<summary><strong>Show Answer</strong></summary>

Measure of data dispersion around the mean.

</details>

### What is a normal distribution?
<details>
<summary><strong>Show Answer</strong></summary>

A symmetric, bell-shaped distribution.

</details>

### What is the mean of a dataset?
<details>
<summary><strong>Show Answer</strong></summary>

Sum of all values divided by the number of values.

</details>

### What is a p-value?
<details>
<summary><strong>Show Answer</strong></summary>

The probability of obtaining test results at least as extreme as the observed results.

</details>


# More Unique Data Analytics Assessment Questions

## Python (General)
### What is a dictionary in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
my_dict = {'a': 1, 'b': 2}
print(my_dict['a'])  # Output: 1
```

A collection of key-value pairs.

</details>

### How do you define a function in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
def greet(name):
    return f"Hello, {name}!"
print(greet("Alice"))  # Output: Hello, Alice!
```

Using the def keyword.

</details>

### What is the use of 'self' in class methods?
<details>
<summary><strong>Show Answer</strong></summary>

```python
class MyClass:
    def __init__(self, value):
        self.value = value
```

Refers to the instance of the class.

</details>

### How do you open a file in read mode?
<details>
<summary><strong>Show Answer</strong></summary>

```python
with open('filename.txt', 'r') as f:
    content = f.read()
```

Using open('filename', 'r')

</details>

### What does the 'range' function do?
<details>
<summary><strong>Show Answer</strong></summary>

```python
for i in range(3):
    print(i)  # Output: 0 1 2
```

Generates a sequence of numbers.

</details>

## NumPy
### What does np.linspace do?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.linspace(0, 1, 5)
print(arr)  # Output: [0.   0.25 0.5  0.75 1.  ]
```

Returns evenly spaced numbers over a specified interval.

</details>

### How to transpose a NumPy array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([[1, 2], [3, 4]])
print(arr.T)
```

Using array.T

</details>

### How to find the maximum in an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 5, 3])
print(np.max(arr))  # Output: 5
```

Using np.max()

</details>

### How to compute the dot product of two arrays?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([1, 2])
b = np.array([3, 4])
print(np.dot(a, b))  # Output: 11
```

np.dot(array1, array2)

</details>

### How to create an array of evenly spaced values?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.arange(0, 5, 1)
print(arr)  # Output: [0 1 2 3 4]
```

Using np.arange()

</details>

## Statistics & Probability
### What is variance?
<details>
<summary><strong>Show Answer</strong></summary>

The average of squared differences from the mean.

</details>

### What is a histogram?
<details>
<summary><strong>Show Answer</strong></summary>

A graphical representation of data distribution.

</details>

### What is the null hypothesis?
<details>
<summary><strong>Show Answer</strong></summary>

A statement that there is no effect or difference.

</details>

### What is a quartile?
<details>
<summary><strong>Show Answer</strong></summary>

One of the three points that divide data into four equal parts.

</details>

### What is a confidence interval?
<details>
<summary><strong>Show Answer</strong></summary>

A range that likely contains a population parameter.

</details>

## Mathematics
### What is the slope of the line y = 3x + 2?
<details>
<summary><strong>Show Answer</strong></summary>

3

</details>

### Convert 45 degrees to radians.
<details>
<summary><strong>Show Answer</strong></summary>

π/4

</details>

### Simplify: (x^2 - 4)/(x - 2)
<details>
<summary><strong>Show Answer</strong></summary>

x + 2

</details>

### What is the result of 5! (factorial)?
<details>
<summary><strong>Show Answer</strong></summary>

120

</details>

### What is the square root of 144?
<details>
<summary><strong>Show Answer</strong></summary>

12

</details>

## Pandas
### How to get summary statistics?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.describe())
```

df.describe()

</details>

### How to sort a DataFrame by a column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df_sorted = df.sort_values('column_name')
```

df.sort_values('column_name')

</details>

### How to rename columns in a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.rename(columns={'old': 'new'})
```

df.rename(columns={'old': 'new'})

</details>

### How to fill missing values with 0?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.fillna(0)
```

df.fillna(0)

</details>

### How to drop missing values in a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.dropna()
```

df.dropna()

</details>

## Data Analysis Concepts
### What is data wrangling?
<details>
<summary><strong>Show Answer</strong></summary>

The process of cleaning and unifying messy data.

</details>

### Why is data visualization important?
<details>
<summary><strong>Show Answer</strong></summary>

To understand data distribution and patterns.

</details>

### What is feature engineering?
<details>
<summary><strong>Show Answer</strong></summary>

Creating new features from existing data.

</details>

### What is data transformation?
<details>
<summary><strong>Show Answer</strong></summary>

Changing the format, structure, or values of data.

</details>

### What is an outlier?
<details>
<summary><strong>Show Answer</strong></summary>

A data point significantly different from others.

</details>


# More Interview Questions

## Python
### What is a Python list?
<details>
<summary><strong>Show Answer</strong></summary>

```python
my_list = [1, 2, 3]
print(my_list[0])  # Output: 1
```

An ordered, mutable collection of items.

</details>

### How do you create a dictionary in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
my_dict = {'key': 'value'}
print(my_dict['key'])  # Output: value
```

Using curly braces with key-value pairs.

</details>

### How do you create a class?
<details>
<summary><strong>Show Answer</strong></summary>

```python
class MyClass:
    pass
```

Using the class keyword.

</details>

### How do you install a Python package?
<details>
<summary><strong>Show Answer</strong></summary>

```bash
pip install package_name
```

Using pip install package_name.

</details>

### What is a context manager?
<details>
<summary><strong>Show Answer</strong></summary>

```python
with open('file.txt', 'r') as f:
    data = f.read()
```

An object that uses __enter__ and __exit__ for resource management.

</details>

### What does 'enumerate' do?
<details>
<summary><strong>Show Answer</strong></summary>

```python
for idx, val in enumerate(['a', 'b']):
    print(idx, val)
# Output: 0 a
#         1 b
```

Returns an iterator of index and value pairs.

</details>

### What is the use of 'self' in a class?
<details>
<summary><strong>Show Answer</strong></summary>

```python
class Example:
    def __init__(self, value):
        self.value = value
```

It refers to the instance of the class.

</details>

### How do you read a file in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
with open('file.txt', 'r') as f:
    content = f.read()
```

Using open('filename', 'r') and read() or readline().

</details>

### What is a lambda function?
<details>
<summary><strong>Show Answer</strong></summary>

```python
square = lambda x: x * x
print(square(4))  # Output: 16
```

An anonymous function defined with the lambda keyword.

</details>

### How do you define a function in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
def add(a, b):
    return a + b
```

Using the def keyword.

</details>

### What is polymorphism?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(len([1, 2, 3]))  # Output: 3
print(len("abc"))      # Output: 3
```

The ability to use a unified interface to operate on different types.

</details>

### What is a tuple?
<details>
<summary><strong>Show Answer</strong></summary>

```python
my_tuple = (1, 2, 3)
```

An ordered, immutable collection of items.

</details>

### What is the difference between '==' and 'is'?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = [1, 2]
b = [1, 2]
print(a == b)  # True
print(a is b)  # False
```

'==' compares values, 'is' compares identities.

</details>

### What are *args and **kwargs?
<details>
<summary><strong>Show Answer</strong></summary>

```python
def func(*args, **kwargs):
    print(args, kwargs)
```

*args is for variable-length positional args, **kwargs for keyword args.

</details>

### What is a decorator?
<details>
<summary><strong>Show Answer</strong></summary>

```python
def my_decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper
```

A function that modifies another function.

</details>

### How do you import a module?
<details>
<summary><strong>Show Answer</strong></summary>

```python
import math
```

Using the import keyword.

</details>

### What is encapsulation?
<details>
<summary><strong>Show Answer</strong></summary>

```python
class MyClass:
    def __init__(self):
        self._hidden = 42
```

Restricting access to some components of an object.

</details>

### What is a set in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
my_set = {1, 2, 3}
```

An unordered collection of unique elements.

</details>

### What does the 'with' statement do when working with files?
<details>
<summary><strong>Show Answer</strong></summary>

```python
with open('file.txt', 'r') as f:
    data = f.read()
```

It ensures proper acquisition and release of resources.

</details>

### How do you handle exceptions in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Error!")
```

Using try-except blocks.

</details>

### What is a Python package vs a module?
<details>
<summary><strong>Show Answer</strong></summary>

A module is a .py file, a package is a folder with __init__.py.

</details>

### What is a Python iterator?
<details>
<summary><strong>Show Answer</strong></summary>

```python
it = iter([1, 2, 3])
print(next(it))  # Output: 1
```

An object with __iter__() and __next__() methods.

</details>

### What is a generator?
<details>
<summary><strong>Show Answer</strong></summary>

```python
def gen():
    yield 1
    yield 2
for x in gen():
    print(x)
```

A function that yields values one at a time using 'yield'.

</details>

### What is duck typing?
<details>
<summary><strong>Show Answer</strong></summary>

An object’s suitability is determined by presence of methods/attributes.

</details>

### What are Python’s data types?
<details>
<summary><strong>Show Answer</strong></summary>

int, float, str, list, tuple, dict, set, bool.

</details>

### What is the use of the 'zip' function?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = [1, 2]
b = ['x', 'y']
print(list(zip(a, b)))  # Output: [(1, 'x'), (2, 'y')]
```

To pair elements from multiple iterables.

</details>

### What is the purpose of 'return' in a function?
<details>
<summary><strong>Show Answer</strong></summary>

```python
def f():
    return 42
```

To send a result back to the caller.

</details>

### What is the difference between 'append' and 'extend'?
<details>
<summary><strong>Show Answer</strong></summary>

```python
lst = [1, 2]
lst.append(3)      # [1, 2, 3]
lst.extend([4, 5]) # [1, 2, 3, 4, 5]
```

'append' adds one item, 'extend' adds multiple items.

</details>

### What is slicing in Python?
<details>
<summary><strong>Show Answer</strong></summary>

```python
lst = [0, 1, 2, 3]
print(lst[1:3])  # Output: [1, 2]
```

Extracting a portion of a sequence.

</details>

### What is a list comprehension?
<details>
<summary><strong>Show Answer</strong></summary>

```python
squares = [x*x for x in range(3)]
print(squares)  # Output: [0, 1, 4]
```

A concise way to create lists using a single line of code.

</details>



## NumPy

### How do you create a NumPy array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
import numpy as np
arr = np.array([1, 2, 3])
```

Using np.array().

</details>

### How to generate an array of zeros?
<details>
<summary><strong>Show Answer</strong></summary>

```python
zeros = np.zeros((2, 2))
```

Using np.zeros(shape).

</details>

### How to generate an array of ones?
<details>
<summary><strong>Show Answer</strong></summary>

```python
ones = np.ones((2, 2))
```

Using np.ones(shape).

</details>

### How to create a sequence of numbers in NumPy?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.arange(0, 5, 1)
```

Using np.arange(start, stop, step).

</details>

### How to create an array of evenly spaced numbers?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.linspace(0, 1, 5)
```

Using np.linspace(start, stop, num).

</details>

### How to reshape a NumPy array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.arange(6).reshape((2, 3))
```

Using array.reshape(new_shape).

</details>

### How to flatten a multi-dimensional array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([[1, 2], [3, 4]])
flat = arr.flatten()
```

Using array.flatten().

</details>

### What does array.T do?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([[1, 2], [3, 4]])
print(arr.T)
```

Returns the transpose of the array.

</details>

### How to get the shape of an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([[1, 2], [3, 4]])
print(arr.shape)
```

Using array.shape.

</details>

### How to get the number of dimensions?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([[1, 2], [3, 4]])
print(arr.ndim)
```

Using array.ndim.

</details>

### How to check the data type of array elements?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 3])
print(arr.dtype)
```

Using array.dtype.

</details>

### How to change the data type of array elements?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 3])
arr = arr.astype(float)
```

Using array.astype(new_type).

</details>

### How to get the size (total number of elements)?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([[1, 2], [3, 4]])
print(arr.size)
```

Using array.size.

</details>

### How to access a specific element?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([[1, 2], [3, 4]])
print(arr[0, 1])  # Output: 2
```

Using indexing like array[i, j].

</details>

### How to slice a NumPy array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([0, 1, 2, 3])
print(arr[1:3])  # Output: [1 2]
```

Using standard Python slicing syntax.

</details>

### How to perform element-wise addition?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([1, 2])
b = np.array([3, 4])
print(a + b)  # Output: [4 6]
```

Using + operator or np.add().

</details>

### How to perform element-wise multiplication?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([1, 2])
b = np.array([3, 4])
print(a * b)  # Output: [3 8]
```

Using * operator or np.multiply().

</details>

### How to compute the dot product?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([1, 2])
b = np.array([3, 4])
print(np.dot(a, b))  # Output: 11
```

Using np.dot(a, b).

</details>

### How to compute matrix multiplication?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.matmul(a, b))
```

Using np.matmul(a, b) or a @ b.

</details>

### How to compute the mean?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 3])
print(np.mean(arr))
```

Using np.mean(array).

</details>

### How to compute the median?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 3, 2])
print(np.median(arr))
```

Using np.median(array).

</details>

### How to compute the standard deviation?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 3])
print(np.std(arr))
```

Using np.std(array).

</details>

### How to find the maximum value?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 5, 3])
print(np.max(arr))
```

Using np.max(array).

</details>

### How to find the minimum value?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 5, 3])
print(np.min(arr))
```

Using np.min(array).

</details>

### How to find the index of the maximum value?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 5, 3])
print(np.argmax(arr))
```

Using np.argmax(array).

</details>

### How to find the index of the minimum value?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 5, 3])
print(np.argmin(arr))
```

Using np.argmin(array).

</details>

### How to sort an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([3, 1, 2])
print(np.sort(arr))
```

Using np.sort(array).

</details>

### How to concatenate two arrays?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([1, 2])
b = np.array([3, 4])
print(np.concatenate([a, b]))
```

Using np.concatenate([a, b]).

</details>

### How to stack arrays vertically?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([1, 2])
b = np.array([3, 4])
print(np.vstack([a, b]))
```

Using np.vstack([a, b]).

</details>

### How to stack arrays horizontally?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([[1], [2]])
b = np.array([[3], [4]])
print(np.hstack([a, b]))
```

Using np.hstack([a, b]).

</details>

### How to create an identity matrix?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(np.eye(3))
```

Using np.eye(n).

</details>

### How to generate random numbers?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(np.random.rand(2, 2))
```

Using np.random module functions.

</details>

### How to set a random seed?
<details>
<summary><strong>Show Answer</strong></summary>

```python
np.random.seed(42)
```

Using np.random.seed(value).

</details>

### What is broadcasting in NumPy?
<details>
<summary><strong>Show Answer</strong></summary>

```python
a = np.array([1, 2, 3])
b = 2
print(a + b)
```

A method to perform operations on arrays of different shapes.

</details>

### How to filter an array by a condition?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, -2, 3])
print(arr[arr > 0])
```

Using boolean indexing like array[array > 0].

</details>

### How to check for NaN values?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, np.nan, 3])
print(np.isnan(arr))
```

Using np.isnan(array).

</details>

### How to replace NaN with a value?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, np.nan, 3])
print(np.nan_to_num(arr))
```

Using np.nan_to_num(array).

</details>

### How to save a NumPy array to a file?
<details>
<summary><strong>Show Answer</strong></summary>

```python
np.save('my_array.npy', arr)
```

Using np.save(filename, array).

</details>

### How to load a NumPy array from a file?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.load('my_array.npy')
```

Using np.load(filename).

</details>

### How to generate a diagonal matrix?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(np.diag([1, 2, 3]))
```

Using np.diag([values]).

</details>

### How to repeat elements of an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2])
print(np.repeat(arr, 2))
```

Using np.repeat(array, repeats).

</details>

### How to tile an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2])
print(np.tile(arr, 3))
```

Using np.tile(array, reps).

</details>

### How to compute the cumulative sum?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 3])
print(np.cumsum(arr))
```

Using np.cumsum(array).

</details>

### How to compute the cumulative product?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 3])
print(np.cumprod(arr))
```

Using np.cumprod(array).

</details>

### How to compute the variance?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 3])
print(np.var(arr))
```

Using np.var(array).

</details>

### How to round values to a certain number of decimals?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1.123, 2.345])
print(np.round(arr, decimals=1))
```

Using np.round(array, decimals=n).

</details>

### How to clip values in an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 5, 10])
print(np.clip(arr, 2, 8))
```

Using np.clip(array, min, max).

</details>

### How to get unique values from an array?
<details>
<summary><strong>Show Answer</strong></summary>

```python
arr = np.array([1, 2, 2, 3])
print(np.unique(arr))
```

Using np.unique(array).

</details>

## Pandas

### How do you create a DataFrame in Pandas?
<details>
<summary><strong>Show Answer</strong></summary>

```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
```

Using pd.DataFrame().

</details>

### How to read a CSV file using Pandas?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = pd.read_csv('filename.csv')
```

Using pd.read_csv('filename.csv').

</details>

### How to read an Excel file?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = pd.read_excel('filename.xlsx')
```

Using pd.read_excel('filename.xlsx').

</details>

### How to view the first 5 rows of a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.head())
```

Using df.head().

</details>

### How to view the last 5 rows of a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.tail())
```

Using df.tail().

</details>

### How to check the shape of a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.shape)
```

Using df.shape.

</details>

### How to get column names?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.columns)
```

Using df.columns.

</details>

### How to get index labels?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.index)
```

Using df.index.

</details>

### How to describe the statistics of a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.describe())
```

Using df.describe().

</details>

### How to check for missing values?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.isnull())
```

Using df.isnull().

</details>

### How to drop missing values?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.dropna()
```

Using df.dropna().

</details>

### How to fill missing values?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.fillna(0)
```

Using df.fillna(value).

</details>

### How to rename DataFrame columns?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.rename(columns={'old': 'new'})
```

Using df.rename(columns={'old': 'new'}).

</details>

### How to select a column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
col = df['column_name']
```

Using df['column_name'].

</details>

### How to select multiple columns?
<details>
<summary><strong>Show Answer</strong></summary>

```python
cols = df[['col1', 'col2']]
```

Using df[['col1', 'col2']].

</details>

### How to filter rows based on condition?
<details>
<summary><strong>Show Answer</strong></summary>

```python
filtered = df[df['col'] > 10]
```

Using df[df['col'] > 10].

</details>

### How to sort a DataFrame by column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df_sorted = df.sort_values('column')
```

Using df.sort_values('column').

</details>

### How to reset the index?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.reset_index()
```

Using df.reset_index().

</details>

### How to set a column as index?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.set_index('column')
```

Using df.set_index('column').

</details>

### How to group data in Pandas?
<details>
<summary><strong>Show Answer</strong></summary>

```python
grouped = df.groupby('column')
```

Using df.groupby('column').

</details>

### How to apply aggregation functions?
<details>
<summary><strong>Show Answer</strong></summary>

```python
agg = df.agg({'col': 'mean'})
```

Using .agg({'col': 'mean'}).

</details>

### How to apply a custom function?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.apply(lambda x: x*2)
```

Using df.apply(func).

</details>

### How to apply a function to each element?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.applymap(lambda x: x*2)
```

Using df.applymap(func).

</details>

### How to concatenate DataFrames?
<details>
<summary><strong>Show Answer</strong></summary>

```python
pd.concat([df1, df2])
```

Using pd.concat([df1, df2]).

</details>

### How to merge two DataFrames?
<details>
<summary><strong>Show Answer</strong></summary>

```python
pd.merge(df1, df2, on='key')
```

Using pd.merge(df1, df2, on='key').

</details>

### How to join DataFrames?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df1.join(df2)
```

Using df1.join(df2).

</details>

### How to drop a column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.drop('column', axis=1)
```

Using df.drop('column', axis=1).

</details>

### How to drop a row?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.drop(index)
```

Using df.drop(index).

</details>

### How to create a new column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df['new'] = [1, 2, 3]
```

Using df['new'] = values.

</details>

### How to map values in a column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df['col'] = df['col'].map({1: 'A', 2: 'B'})
```

Using df['col'].map(mapping).

</details>

### How to replace values in a column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df['col'] = df['col'].replace(1, 100)
```

Using df['col'].replace(old, new).

</details>

### How to check data types of columns?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.dtypes)
```

Using df.dtypes.

</details>

### How to change data type of a column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df['col'] = df['col'].astype(float)
```

Using df['col'].astype(type).

</details>

### How to read a specific number of rows from a file?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = pd.read_csv('file.csv', nrows=100)
```

Using pd.read_csv(..., nrows=100).

</details>

### How to write a DataFrame to CSV?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df.to_csv('file.csv')
```

Using df.to_csv('file.csv').

</details>

### How to write a DataFrame to Excel?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df.to_excel('file.xlsx')
```

Using df.to_excel('file.xlsx').

</details>

### How to check for duplicate rows?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.duplicated())
```

Using df.duplicated().

</details>

### How to drop duplicate rows?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = df.drop_duplicates()
```

Using df.drop_duplicates().

</details>

### How to create a DataFrame from a dictionary?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df = pd.DataFrame({'col': [1, 2]})
```

Using pd.DataFrame({'col': [1, 2]}).

</details>

### How to pivot a table?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df.pivot(index='row', columns='col', values='val')
```

Using df.pivot(index, columns, values).

</details>

### How to melt a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
pd.melt(df)
```

Using pd.melt(df).

</details>

### How to get unique values in a column?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df['col'].unique())
```

Using df['col'].unique().

</details>

### How to count unique values?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df['col'].nunique())
```

Using df['col'].nunique().

</details>

### How to count value frequencies?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df['col'].value_counts())
```

Using df['col'].value_counts().

</details>

### How to sample rows from a DataFrame?
<details>
<summary><strong>Show Answer</strong></summary>

```python
df.sample(n=5)
```

Using df.sample(n=5).

</details>

### How to check memory usage?
<details>
<summary><strong>Show Answer</strong></summary>

```python
print(df.memory_usage())
```

Using df.memory_usage().

</details>

### How to iterate over rows?
<details>
<summary><strong>Show Answer</strong></summary>

```python
for idx, row in df.iterrows():
    print(idx, row)
```

Using df.iterrows().

</details>

### How to check if a column exists?
<details>
<summary><strong>Show Answer</strong></summary>

```python
'col' in df.columns
```

'col' in df.columns.

</details>



## Mathematics

### What is the derivative of x^2?
<details>
<summary><strong>Show Answer</strong></summary>

2x

</details>

### What is the integral of x dx?
<details>
<summary><strong>Show Answer</strong></summary>

0.5 * x^2 + C

</details>

### What is the value of π (pi) to 3 decimal places?
<details>
<summary><strong>Show Answer</strong></summary>

3.142

</details>

### What is the Pythagorean theorem?
<details>
<summary><strong>Show Answer</strong></summary>

a^2 + b^2 = c^2

</details>

### What is the formula for the area of a circle?
<details>
<summary><strong>Show Answer</strong></summary>

π * r^2

</details>

### What is the formula for the circumference of a circle?
<details>
<summary><strong>Show Answer</strong></summary>

2 * π * r

</details>

### What is the quadratic formula?
<details>
<summary><strong>Show Answer</strong></summary>

x = (-b ± √(b^2 - 4ac)) / 2a

</details>

### What is the slope-intercept form of a line?
<details>
<summary><strong>Show Answer</strong></summary>

y = mx + b

</details>

### What is the factorial of 5?
<details>
<summary><strong>Show Answer</strong></summary>

120

</details>

### What is 2 to the power of 5?
<details>
<summary><strong>Show Answer</strong></summary>

32

</details>

### What is the square root of 81?
<details>
<summary><strong>Show Answer</strong></summary>

9

</details>

### What is the standard deviation?
<details>
<summary><strong>Show Answer</strong></summary>

The measure of the spread of a dataset.

</details>

### What is the mean of 2, 4, 6, 8?
<details>
<summary><strong>Show Answer</strong></summary>

5

</details>

### What is the median of 1, 3, 3, 6, 7, 8, 9?
<details>
<summary><strong>Show Answer</strong></summary>

6

</details>

### What is the mode of 1, 2, 2, 3, 4?
<details>
<summary><strong>Show Answer</strong></summary>

2

</details>

### What is a matrix?
<details>
<summary><strong>Show Answer</strong></summary>

A rectangular array of numbers.

</details>

### What is matrix multiplication?
<details>
<summary><strong>Show Answer</strong></summary>

Row-by-column multiplication of two matrices.

</details>

### What is the identity matrix?
<details>
<summary><strong>Show Answer</strong></summary>

A square matrix with 1s on the diagonal and 0s elsewhere.

</details>

### What is a determinant?
<details>
<summary><strong>Show Answer</strong></summary>

A scalar value derived from a square matrix.

</details>

### What is a vector?
<details>
<summary><strong>Show Answer</strong></summary>

A quantity with both magnitude and direction.

</details>

### What is a scalar?
<details>
<summary><strong>Show Answer</strong></summary>

A single value without direction.

</details>

### What is a logarithm?
<details>
<summary><strong>Show Answer</strong></summary>

The inverse operation of exponentiation.

</details>

### What is log(100) base 10?
<details>
<summary><strong>Show Answer</strong></summary>

2

</details>

### What is e (Euler's number)?
<details>
<summary><strong>Show Answer</strong></summary>

Approximately 2.718

</details>

### What is the derivative of sin(x)?
<details>
<summary><strong>Show Answer</strong></summary>

cos(x)

</details>

### What is the integral of 1/x dx?
<details>
<summary><strong>Show Answer</strong></summary>

ln|x| + C

</details>

### What is the value of sin(90°)?
<details>
<summary><strong>Show Answer</strong></summary>

1

</details>

### What is the cosine of 0°?
<details>
<summary><strong>Show Answer</strong></summary>

1

</details>

### What is tan(45°)?
<details>
<summary><strong>Show Answer</strong></summary>

1

</details>

### What is the binomial coefficient formula?
<details>
<summary><strong>Show Answer</strong></summary>

n! / [r!(n - r)!]

</details>

### What is a permutation?
<details>
<summary><strong>Show Answer</strong></summary>

An arrangement of items in a specific order.

</details>

### What is a combination?
<details>
<summary><strong>Show Answer</strong></summary>

A selection of items regardless of order.

</details>

### What is variance?
<details>
<summary><strong>Show Answer</strong></summary>

The average of the squared differences from the mean.

</details>

### What is a probability?
<details>
<summary><strong>Show Answer</strong></summary>

A measure of the likelihood of an event.

</details>

### What is a normal distribution?
<details>
<summary><strong>Show Answer</strong></summary>

A symmetric bell-shaped distribution.

</details>

### What is a standard normal distribution?
<details>
<summary><strong>Show Answer</strong></summary>

A normal distribution with mean 0 and SD 1.

</details>

### What is a z-score?
<details>
<summary><strong>Show Answer</strong></summary>

The number of SDs from the mean.

</details>

### What is a linear equation?
<details>
<summary><strong>Show Answer</strong></summary>

An equation that graphs a straight line.

</details>

### What is the inverse of a function?
<details>
<summary><strong>Show Answer</strong></summary>

A function that reverses the original function.

</details>

### What is the domain of a function?
<details>
<summary><strong>Show Answer</strong></summary>

All possible input values.

</details>

### What is the range of a function?
<details>
<summary><strong>Show Answer</strong></summary>

All possible output values.

</details>

### What is an arithmetic sequence?
<details>
<summary><strong>Show Answer</strong></summary>

A sequence with a constant difference between terms.

</details>

### What is a geometric sequence?
<details>
<summary><strong>Show Answer</strong></summary>

A sequence with a constant ratio between terms.

</details>

### What is the nth term of an arithmetic sequence?
<details>
<summary><strong>Show Answer</strong></summary>

a + (n-1)d

</details>

### What is the nth term of a geometric sequence?
<details>
<summary><strong>Show Answer</strong></summary>

a * r^(n-1)

</details>

### What is a limit in calculus?
<details>
<summary><strong>Show Answer</strong></summary>

The value a function approaches as input approaches a point.

</details>

### What is continuity?
<details>
<summary><strong>Show Answer</strong></summary>

A function with no breaks, holes, or jumps.

</details>

### What is a piecewise function?
<details>
<summary><strong>Show Answer</strong></summary>

A function defined by multiple expressions for different intervals.

</details>

### What is an asymptote?
<details>
<summary><strong>Show Answer</strong></summary>

A line that a graph approaches but never touches.

</details>

