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


