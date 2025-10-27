
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df = pd.read_csv('car_data.csv')
df.head()

df.info()

df.isnull().sum()

df.columns

df.describe()

df['Year'].unique()

df['Owner'].unique()

df['Age'] =  2025 - df['Year']
df.head()

checking_df= df.drop(['Car_Name','Transmission','Fuel_Type','Seller_Type','Year'],axis=1)

checking.head()
sns.heatmap(checking_df.corr(),annot=True)
plt.show()

sns.pairplot(df, vars=['Selling_Price','Year','Kms_Driven'])

sns.boxplot(x=df["Selling_Price"])

encoded_cols = pd.get_dummies(df[['Fuel_Type', 'Seller_Type', 'Transmission']], drop_first=True)

df_encoded = pd.concat([df, encoded_cols], axis=1)

df_encoded = df_encoded.drop(['Fuel_Type', 'Seller_Type', 'Transmission', 'Car_Name','Year'], axis=1)
df_encoded.head()

cols = ['Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual', 'Transmission_Manual']
df_encoded[cols] = df_encoded[cols].astype(int)
df_encoded.head()

df_encoded.corr(numeric_only=True)['Selling_Price'].sort_values(ascending=False)

X = df_encoded.drop('Selling_Price', axis=1)
y = df_encoded['Selling_Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
def correlation_for_dropping(df_encoded, threshold):
    columns_to_drop = set()
    corr = df_encoded.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                columns_to_drop.add(corr.columns[i])
    return columns_to_drop

columns_dropping = correlation_for_dropping(X_train, 0.95)
print(columns_dropping)

print(X_train.corr().iloc[:, 1].sort_values(ascending=False))

print(df_encoded[['Fuel_Type_Diesel','Fuel_Type_Petrol']].corr())

X_train = X_train.drop(columns=['Fuel_Type_Petrol'], axis=1)
X_test = X_test.drop(columns=['Fuel_Type_Petrol'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title("X_train")
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title("X_train_scaled")
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_pred = linear.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", score)
plt.scatter(y_test, y_pred)
plt.show()

sample = pd.DataFrame({
    'Present_Price': [9.85],
    'Kms_Driven': [6900],
    'Owner': [0],
    'Age': [8],
    'Fuel_Type_Diesel': [0],
    'Seller_Type_Individual': [0],
    'Transmission_Manual': [1]
})
sample_scaled = scaler.transform(sample)
predicted_price = linear.predict(sample_scaled)
print("Selling prediction of sample:  ", predicted_price[0])
