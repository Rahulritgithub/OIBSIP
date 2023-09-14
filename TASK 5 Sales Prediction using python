import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('Advertising.csv')

df.head()

df.tail()

df.info()

df.describe()

df.isnull().sum()

fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(df['TV'], ax = axs[0])
plt2 = sns.boxplot(df['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(df['Radio'], ax = axs[2])
plt.tight_layout()

sns.distplot(df['Newspaper'])

iqr = df.Newspaper.quantile(0.75) - df.Newspaper.quantile(0.25)

lower_bridge = df["Newspaper"].quantile(0.25) - (iqr*1.5)
upper_bridge = df["Newspaper"].quantile(0.75) - (iqr*1.5)
print(lower_bridge)
print(upper_bridge)

data=df.copy()
data.loc[data['Newspaper']>=93, 'Newspaper']=93
sns.boxplot(data['Sales']);

sns.heatmap(data.corr(), cmap="YlGnBu", annot = True)
plt.show

im_feat = list(df.corr()['Sales'][(df.corr()['Sales']>+0.5)|(df.corr()['Sales']<-0.5)].index)

print(im_feat)

x=data['TV']
y=data['Sales']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.33)

print(x_train.shape, y_train.shape)

knn = KNeighborsRegressor().fit(x_train,y_train)
knn

knn_train_pred = knn.predict(x_train)
knn_test_pred = knn.predict(x_test)
print(knn_train_pred, knn_test_pred)

results = pd.DataFrame(columns=["Model","Train R2","Test R2","Test RMSE","Variance"])

r2 =  r2_score(y_test,knn_test_pred)
r2_train = r2_score(y_train,knn_train_pred)
rmse = np.sqrt(mean_squared_error(y_test,knn_test_pred))
Variance = r2_train - r2
results = results.append({"Model":"K-Nearest Neighbors","Train R2":r2_train,"Test R2":r2,"Test RMSE":rmse,"Variance":Variance},ignore_index=True)
print("R2:",r2)
print("RMSE:",rmse)

results.head()
