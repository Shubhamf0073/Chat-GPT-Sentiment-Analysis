from nltk.util import pr 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
import re 
import nltk
nltk.download()
stemmmer = nltk.SnowballStemmer('english')
from nltk.corpus import stopwords 
import string 
stopwords = set(stopwords.words('english')) 
nltk.download('stopwords')
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_score, accuracy_score, f1_score, classification_report 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import plotly.express as px 
import seaborn as sns  
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv(r'C:\Users\Lenovo\Downloads\archive (2)\file.csv')
data.head()
data.tail()
data.shape
data.info()
data.isnull().sum()
data.duplicated().sum()
data.columns
del data['Unnamed: 0']

# Cleaning the Dataset
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    text = [stemmmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text
data["tweets"] = data["tweets"].apply(clean)

# Checking Labels

data['labels'].value_counts()

data['labels'] = data['labels'].astype(str)
data['labels'] = data['labels'].str.strip()
data['labels'].value_counts().plot(kind = 'pie', autopct = '%.2f')
labels = data['labels'].value_counts()
numbers = labels.index
quantity = labels.values
figure = px.pie(data, values = quantity,  names =  numbers, hole = 0.5)
figure.show()

text = ' '.join(i for i in data.labels)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords = stopwords, background_color = 'white').generate(text)
plt.figure(figsize = (5,5))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

text = " ".join(i for i in data.tweets)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords = stopwords, background_color = 'White').generate(text)
plt.figure(figsize = (5,5))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

# Spliting the Dataset
X = data['tweets']
Y = data['labels']
# Loading CounterVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)
print(X)

le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)

lg = LogisticRegression()
lg.fit(X_train, Y_train)

# Training dataset accuracy score
test_data = lg.predict(X_test)
accuracy_score(test_data, Y_test)

preds = lg.predict(X_test)
print(classification_report(Y_test, preds)) 

DT = DecisionTreeClassifier()
# Fitting the Model
DT.fit(X_train, Y_train)

# Accuracy for training Dataset
train_test = DT.predict(X_train)
accuracy_score(train_test, Y_train)

# Accuracy for Test Dataset
test_data = DT.predict(X_test)
accuracy_score(test_data, Y_test)

preds = DT.predict(X_test)
print(classification_report(Y_test, preds))

nb = MultinomialNB()
nb.fit(X_train, Y_train)
preds = nb.predict(X_test)
print(classification_report(Y_test, preds))

# HPT for Multinnomial Naive Bayes Model
param_grid = {'alpha': [0.1, 0, 1.0, 10, 100]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, verbose = 2)
grid_search.fit(X_train, Y_train)

grid_search.best_params_
