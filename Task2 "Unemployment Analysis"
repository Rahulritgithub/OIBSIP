import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv("Unemployment in India.csv")
df.head()

df.tail()

df.shape

df.describe()

px.bar(df,x="Region")

px.bar(df,x="Region", color = "Region")

data = df.groupby(["Region"]).size().rename("Count").reset_index()
data.head()

df.head()

px.bar(data,x="Region", y="Count",color="Region",text="Count")

fig = px.box(df,x='Region', y=' Estimated Unemployment Rate (%)',color='Region',title='Unemp rate', animation_frame = ' Date',template = 'plotly')
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.show()

fig = px.scatter(df,x='Region', y=' Estimated Unemployment Rate (%)',color='Region',title='Unemp rate', animation_frame = ' Date',template = 'plotly')
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.show()

df.describe()

df['Region'].value_counts().plot(kind='pie',autopct='%.1f')

fig = px.violin(df,x='Region', y=' Estimated Unemployment Rate (%)',color='Region',box=False,points='all',hover_data=df.columns)
fig.show()
