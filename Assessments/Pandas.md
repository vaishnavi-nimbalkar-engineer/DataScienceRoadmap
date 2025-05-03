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



