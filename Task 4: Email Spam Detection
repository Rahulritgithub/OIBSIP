import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re,string
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('stopwords')
%matplotlib inline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV


df=pd.read_csv("spam.csv",encoding='ISO-8859-1')
df

df.isnull().sum()

df.head()

df.tail()

df = df[['v1','v2']]
df.columns=['label','message']
df.head()

df.info()

df.groupby('label')

df.shape

df['label'].value_counts().plot(kind='bar')

ps=PorterStemmer()
corpus=[]
for i in range(0, len(df)):
  review = re.sub('[^a-zA-Z]',' ',df['message'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=4000)
x=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(df['label'])
y=y.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 42)


m1 = RandomForestClassifier()
m1.fit(x_train, y_train)

m2=DecisionTreeClassifier()
m2.fit(x_train, y_train)

from sklearn.naive_bayes import MultinomialNB
m3=MultinomialNB()
m3.fit(x_train, y_train)

pr1=m1.predict(x_test)
pr2=m2.predict(x_test)
pr3=m3.predict(x_test)
print(pr1)
print(pr2)
print(pr3)

print("Random Forest Classifier")
print("confusion matrix: ")
print(confusion_matrix(y_test,pr1))
print("Accuracy: ",  accuracy_score(y_test,pr1))

print("Decision tree")
print("confusion matrix: ")
print(confusion_matrix(y_test,pr2))
print("Accuracy: ",  accuracy_score(y_test,pr2))

print("Naive Bayes")
print("confusion matrix: ")
print(confusion_matrix(y_test,pr3))
print("Accuracy: ",  accuracy_score(y_test,pr3))

cm = confusion_matrix(y_test, pr3)
sns.heatmap(cm, annot=True)

report1 = classification_report(y_test,pr1)
print("Report for Random forest classification ",report1)
report2 = classification_report(y_test,pr2)
print("Report for decision Tree",report2)
report3 = classification_report(y_test,pr3)
print("Report for Navies bayes",report3)
