import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder

df = pd.read_csv("CarPrice_Assignment.csv")
df.head()

df.tail()

df = df.dropna()
df.info()

df.isnull().sum()

df.describe(include='all')

df["price"].plot.hist()

sns.countplot(x='fueltype',data=df)
plt.title('fuel Type Dist')
plt.show()

plt.scatter(df['enginesize'], df['horsepower'])
plt.xlabel('Engine size')
plt.ylabel('Horse power')
plt.title('Eng size and horse power')
plt.show()

sns.pairplot(df[['price','enginesize','horsepower','curbweight']])
plt.show()

from sklearn import preprocessing
le=LabelEncoder()
var_mod = df.select_dtypes(include='object').columns
for i in var_mod:
  df[i]=le.fit_transform(df[i])

X=df.drop(['price'],axis=1)
y=df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(mse)
print(r2)

import numpy as np


new_car_features = np.array([5000, 0, 2, 3, 0, 86.0, 162.0, 67.4, 2222, 100, 4, 4.46, 4.19, 9.0, 68, 5550, 31, 38, 0, 0, 0, 0, 0, 0, 0])

new_car_features_31 = np.concatenate((new_car_features, np.zeros(6)))

new_car_features_2d = new_car_features_31.reshape(1, -1)

new_car_price = model.predict(new_car_features_2d)
print("predicted price:", new_car_price[0])



import numpy as np


new_car_features = np.array([5000, 0, 2, 3, 0, 86.0, 162.0, 67.4, 2222, 100, 4, 4.46, 4.19, 9.0, 68, 5550, 31, 38, 0, 0, 0, 0, 0, 0, 0])

new_car_features_31 = np.concatenate((new_car_features, np.zeros(6)))

new_car_features_2d = new_car_features_31.reshape(1, -1)

new_car_price = model.predict(new_car_features_2d)
print("predicted price:", new_car_price[0])

#predicted price: 62452200.50437117
