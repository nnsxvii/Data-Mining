import pandas as pd

# 1.1 Calculate fuel efficiency (km/liter) from distance and fuel consumption
df = pd.DataFrame({
    'Distance': [300, 450, 600],  # in kilometers
    'Fuel_Consumed': [15, 20, 25]  # in liters
})

df['Fuel_Efficiency'] = df['Distance'] / df['Fuel_Consumed']
print("Data with Fuel Efficiency Feature:")
print(df)

# -----------------------------------------------------------------------------

# 1.2 Create polynomial features from existing financial data
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({
    'Years_Experience': [1, 2, 3],
    'Salary': [40000, 50000, 60000]
})

poly = PolynomialFeatures(degree=2, include_bias=False)
new_features = poly.fit_transform(df)

new_features_df = pd.DataFrame(new_features, columns=poly.get_feature_names_out(df.columns))
print("\nPolynomial Features DataFrame from Financial Data:")
print(new_features_df)

# -----------------------------------------------------------------------------

# 2. Extract date-based features from project deadlines
df = pd.DataFrame({
    'Project_Deadline': ['2023-11-01', '2024-05-10', '2025-01-20']
})

df['Project_Deadline'] = pd.to_datetime(df['Project_Deadline'])

df['Year'] = df['Project_Deadline'].dt.year
df['Month'] = df['Project_Deadline'].dt.month
df['Day'] = df['Project_Deadline'].dt.day
df['Week_of_Year'] = df['Project_Deadline'].dt.isocalendar().week

print("\nData with Project Deadline Date-Based Features:")
print(df)

# -----------------------------------------------------------------------------

# 3. Engineer features using domain knowledge (e.g., daily commute time)
df = pd.DataFrame({
    'Commute_Start': ['07:30', '09:00', '17:45'],
    'Commute_End': ['08:30', '10:00', '18:30']
})

df['Commute_Start'] = pd.to_datetime(df['Commute_Start'], format='%H:%M').dt.time
df['Commute_End'] = pd.to_datetime(df['Commute_End'], format='%H:%M').dt.time

df['Commute_Duration'] = pd.to_datetime(df['Commute_End'].astype(str), format='%H:%M') - pd.to_datetime(df['Commute_Start'].astype(str), format='%H:%M')
df['Commute_Duration'] = df['Commute_Duration'].dt.total_seconds() / 3600  # Convert duration to hours

def commute_type(hour):
    if 7 <= hour < 10:
        return 'Morning Commute'
    elif 17 <= hour < 19:
        return 'Evening Commute'
    else:
        return 'Off-Hours Commute'

df['Start_Commute_Type'] = df['Commute_Start'].apply(lambda x: commute_type(x.hour))
df['End_Commute_Type'] = df['Commute_End'].apply(lambda x: commute_type(x.hour))

print("\nData with Commute Duration and Commute Type:")
print(df)
