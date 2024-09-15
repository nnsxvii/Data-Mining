# 1. Normalize numerical features using Min-Max scaling
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = pd.DataFrame({
    'salary': [3000, 5000, 7000, 9000, 11000]
})

scaler = MinMaxScaler()

data[['salary']] = scaler.fit_transform(data[['salary']])
print("Normalized Salary Data:")
print(data)

# -----------------------------------------------------------------------------

# 2. Encode categorical variables using OneHotEncoder from sklearn
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame({
    'Department': ['HR', 'Engineering', 'Marketing', 'HR'],
    'Experience': [5, 10, 3, 8]
})

encoder = OneHotEncoder(sparse=False)

encoded_departments = encoder.fit_transform(data[['Department']])

encoded_df = pd.DataFrame(encoded_departments, columns=encoder.get_feature_names_out(['Department']))

data_encoded = pd.concat([data[['Experience']], encoded_df], axis=1)
print("\nOne-Hot Encoded Data (Departments):")
print(data_encoded)

# -----------------------------------------------------------------------------

# 3. Bin continuous variables into discrete intervals using pd.cut()
import pandas as pd

data = pd.DataFrame({
    'Weight': [55, 65, 75, 85, 95, 105, 115]
})

data['Weight_Group'] = pd.cut(data['Weight'], bins=4, labels=['Light', 'Moderate', 'Heavy', 'Very Heavy'])
print("\nBinned Weight Data:")
print(data)
