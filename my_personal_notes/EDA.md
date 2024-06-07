# Data wrangling

## Data visualization and profiling

> Identifying distribution, structure and problems of dataset.

- These are basic commands
```py
# for structure of data:
dataframe.info()

# for basic statistics:
dataframe.describe() # parameter include='O' gives about object type too
```

### Widely used diagrams: 

- Barplot: For frequency of categorical data
- Histogram: For distribution of discrete data
- Boxplot: 5 number summary (again distribution and outliers)
- Scatterplot: relation between two numerical values
- Heatmap: visualization of correlation and missing values

## Data cleaning

### Possible issues:

- With tabular/structured data:
    - Irrelevant features
    - Parsing and type conversion
    - Structural errors

- With text data: (**Remember nltk, text blob and jennison**)
    - Misspelling
    - Random while spacing
    - html tags, xml tags (from scraping)

- With image data: (**Use hosaka plot to inspect**)
    - blur
    - missing pixels
    - over/under exposure
    - noisy image
    - others: distortion, partial obstructions etc.

### Handling missing values

- Deletion : 
- Imputation : Filling using mean, median, mode OR using multivariate imputation