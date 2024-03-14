# install pandas, matplotlib, nltk using pip before running this file
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('text.csv')
df.head()

df.columns

df.isnull().sum()

df.info()

df.shape

sentiment_count = df.label.value_counts()
sentiment_type = df.label.value_counts().index
plt.pie(sentiment_count,labels=sentiment_type, autopct='1.1f%%',colors=['green', 'red'])

import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)

df['label'].value_counts()

import re
from nltk.stem.porter import PorterStemmer

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('^a-zA-z',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return(stemmed_content)

df['stemmed_content'] = df['text'].apply(stemming)
df.head()

print(df['stemmed_content'])
print(df['label'])

# Export DataFrame to CSV
# df.to_csv('output2.csv', index=False)
