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



