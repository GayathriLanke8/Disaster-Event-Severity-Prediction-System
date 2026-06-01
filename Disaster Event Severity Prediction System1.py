import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#  Step 1: Data Collection & Loading

df = pd.read_csv("/content/synthetic_disaster_events_2025.csv")

df.shape

# Step 2: Initial Data Inspection

df.head()

df.tail()

df.info()

df.describe()

df.columns

df['is_major_disaster'].value_counts()

#  Step 3: Data Cleaning

df.isnull().sum()

df.duplicated().sum()

# Step 4: Exploratory Data Analysis (EDA)

num_cols = ["severity_level", "affected_population", "estimated_economic_loss_usd", "response_time_hours", "infrastructure_damage_index"]

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()

df[num_cols].skew()

df['estimated_economic_loss_usd'] = np.log1p(df['estimated_economic_loss_usd'])

features = ['severity_level','affected_population',
            'estimated_economic_loss_usd',
            'response_time_hours']

for col in features:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='is_major_disaster', y=col, data=df)
    plt.title(f"{col} vs Target")
    plt.show()

sns.countplot(x='disaster_type', hue='is_major_disaster', data=df)
plt.xticks(rotation=45)
plt.show()

sns.countplot(x='location', hue='is_major_disaster', data=df)
plt.xticks(rotation=45)
plt.show()

sns.countplot(x='aid_provided', hue='is_major_disaster', data=df)
plt.xticks(rotation=45)
plt.show()

df.select_dtypes(include=['number']).corr()['is_major_disaster']

plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.show()

sns.pairplot(df[['severity_level',
                 'affected_population',
                 'estimated_economic_loss_usd',
                 'is_major_disaster']],
             hue='is_major_disaster')
plt.show()

sns.scatterplot(
    x='affected_population',
    y='estimated_economic_loss_usd',
    hue='is_major_disaster',
    data=df
)
plt.show()

# Step 5: Outlier Detection & Treatment

plt.boxplot(df['affected_population'])
plt.show()

Q1 = df['response_time_hours'].quantile(0.25)
Q3 = df['response_time_hours'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['response_time_hours'] < lower) |
              (df['response_time_hours'] > upper)]

print(outliers)

# Step 6: Feature Encoding

df.drop(columns=['event_id'], inplace=True)

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df.drop(columns=['date'], inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_cols = ['disaster_type', 'location', 'aid_provided']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df.head()

df.info()

# Step 7: Feature Scaling & Step 8: Train-Test Split

from sklearn.model_selection import train_test_split

X = df.drop('is_major_disaster', axis=1)
y = df['is_major_disaster']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
