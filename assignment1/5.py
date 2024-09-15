import pandas as pd

# -----------------------------------------------------------------------------
# 1. Identify and remove duplicate rows in a products dataset
data = {
    'Product': ['Laptop', 'Phone', 'Tablet', 'Laptop', 'Phone', 'Laptop'],
    'Category': ['Electronics', 'Gadgets', 'Electronics', 'Electronics', 'Gadgets', 'Electronics']
}

df = pd.DataFrame(data)

print("Duplicate Rows in the Products DataFrame:")
print(df[df.duplicated()])

df_no_duplicates = df.drop_duplicates(keep='last')
print("\nDataFrame without duplicates (keeping the last occurrence):")
print(df_no_duplicates)

# -----------------------------------------------------------------------------

# 2. Detect and remove outliers in a house price dataset using Z-score
from scipy import stats

data = {
    'Price': [250000, 275000, 300000, 1200000, 290000, 310000, 325000]
}

df = pd.DataFrame(data)

z_scores = stats.zscore(df['Price'])
abs_z_scores = abs(z_scores)

df_no_outliers = df[abs_z_scores < 3]

print("\nHouse Prices DataFrame without outliers:")
print(df_no_outliers)

# -----------------------------------------------------------------------------

# 3. Correct inconsistencies in a location dataset
data = {
    'Location': ['los angeles', 'SAN FRANCISCO', 'Los Angeles', 'san francisco', 'chicago'],
    'State': ['ca', 'CA', 'California', 'california', 'IL']
}

df = pd.DataFrame(data)

df['Location'] = df['Location'].str.title()
df['State'] = df['State'].str.upper()

print("\nStandardized Location DataFrame:")
print(df)
