#**Importing Modules for reading data set and to train models**

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

#**Loading dataset for traning the models and to read**

df = pd.read_csv('Iris.csv')
df.head()

df.describe()

df.info()

#**Preprocessing the data set by filling the null values**

df.isnull().sum()

#Data Analysis in the form of graph

df['SepalLengthCm'].hist()

df['SepalWidthCm'].hist()

df['PetalLengthCm'].hist()

colors = ['blue', 'red', 'orange']
species = ['Iris-virginica', 'Iris-versicolor','Iris-setosa']

for i in range(3):
  x = df[df['Species'] == species[i]]
  plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c =colors[i], label = species[i])
  plt.xlabel("Sepal Length")
  plt.ylabel("Sepal width")
  plt.legend()

for i in range(3):
  x = df[df['Species'] == species[i]]
  plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c =colors[i], label = species[i])
  plt.xlabel("Petal Length")
  plt.ylabel("petal width")
  plt.legend()

#Corr-Matrix shows Correlation Coefficients between Variables


corr=df.corr()
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(corr, annot=True, ax=ax)

#Label Encoder - Converting the labels into numeric form - Convert into Machine Readable Form

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species']= le.fit_transform(df['Species'])
df.head()

#Traning the model

from sklearn.model_selection import train_test_split
X=df.drop(columns=['Species'])
Y=df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

#Using Logistic Regression it is classification model

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

#Testing the Accuracy of the model

print("Accuracy:",model.score(x_test, y_test)*100)
