# Pandas
> Data manipulation tool that we use mostly for DataFrame

`df = pd.read_csv("path/to/file.csv", index_col=0)` - If there some column is the index or pandas assigns itself.

## Index and select data

- Using bracket
```py
df["column"] # Data type will be series (1D-labelled array)
df[["coulumn1", "column2"]] # This returns dataframe
df[3:6] # Returns 4th 5th and 6th row (index 3,4,5)
```

- loc (label-based)
```py
df.loc["row1"] # Panda series
# OR
df.loc[["row1"]] # Returns a DataFrame
# Also
df.loc[["row1", "row2"]]

# For selection based on both row and column
df.loc[["row1", "row2", "row3"], ["column1", "column2"]] # Also a dataframe
df.loc[:, ["column1", "column2"]] # All rows of two columns

# Similar thing can be done for column
```

- iloc (index based)
```py
df.loc[[1]] # Pandas df of 1st row
df.iloc[[1,2,3], [0,1]] # 3 rows, 2 columns
# also
df.iloc[:, [0,1]] # All rows of two columns
```
## Customizing plots

- Labeling Axes and Titles: xlabel(), ylabel(), and title()
```py

```

- Adjusting Ticks: yticks() and xticks()
```py

```

