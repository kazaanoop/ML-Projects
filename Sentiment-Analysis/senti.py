import pandas as pd
import re
import numpy as np
from nltk.stem.porter import *
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.sentiment.util import extract_unigram_feats, mark_negation

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#dropping ID columns
train_id = train['id']
test_id = test['id']
train = train.drop(columns = 'id' , axis =1)
test = test.drop(columns='id', axis=1)

#Splitting the target value
n_train = train.shape[0]
n_test = test.shape[0]
y_train = train['label']
train = train.drop(columns = 'label' ,axis =1 )
data = pd.concat((train,test)).reset_index(drop=True)

#Removing patterns
def rem_pattern(x,pattern):
    r = re.findall(pattern,x)
    for i in r:
        x = re.sub(i,' ',x)

    return x

#Removing twitter handels, punctuations, url's, short words
data['tidy_tweet'] = np.vectorize(rem_pattern)(data['tweet'],"@[\w]*")


nltk.download('vader_lexicon')

sid = SIA()

#new features
#data['tidy_tweet'] =  data['tidy_tweet'].apply(lambda x: x.replace('$&@*#', 'huita' ))
data['nltkdict'] = data['tidy_tweet'].apply(lambda x: sid.polarity_scores(x))
data['nltkcompound'] = data['nltkdict'].apply(lambda x: x['compound'])
data['nltkneg'] = data['nltkdict'].apply(lambda x: x['neg'])
data['nltkpos'] = data['nltkdict'].apply(lambda x: x['pos'])
data['tweets_neg'] = data['tidy_tweet'].apply(lambda x: ' '.join(mark_negation(x.split())))
data['count#'] = data['tidy_tweet'].apply(lambda x: x.count('#'))
data['count!'] = data['tidy_tweet'].apply(lambda x: x.count('!'))
data['countupper'] =  data['tidy_tweet'].apply(lambda x: len([l for l in x if l.isupper()]))
data['countword'] =  data['tidy_tweet'].apply(lambda x: len(x.split()))
data['countletter'] =  data['tidy_tweet'].apply(lambda x: len(x))

data['tidy_tweet'] = data['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
data['tidy_tweet'] = data['tidy_tweet'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))
data['tidy_tweet'] = data['tidy_tweet'].apply(lambda x: x.lower())
#data['tidy_tweet'] = rem_pattern(data['tidy_tweet'],"http[s]?:\/\/.*[\r\n]*")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("http[s]?:\/\/.*[a-zA-Z0-9/]*"," ")
data['tidy_tweet'] = data['tidy_tweet'].apply(lambda x:' '.join([w for w in x.split(' ') if not w.startswith("http")]))

#Removing proper nouns
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#apple"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("apple"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#iphone"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("iphone"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#samsung"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#phone"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("phone"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("instagram"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("twitter"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("instagr"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#sony"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#ipod"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#ipad"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#android"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("ipod"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("#like"," ")
data['tidy_tweet'] = data['tidy_tweet'].str.replace("like"," ")

#Tokenizing
token = data['tidy_tweet'].apply(lambda x: x.split())

#Remove stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
token = token.apply(lambda x: [item for item in x if item not in stop_words])

#Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
token = token.apply(lambda x:[lemmatizer.lemmatize(i) for i in x])

for i in range(len(token)):
    token[i] = ' '.join(token[i])

data['tidy_tweet'] = token

#removing useless words
#useless = ['#apple','apple','#phone','#iphone','phone','iphone','samsung','twitter','sony','ipod','ipad']
#token = token.apply(lambda x: [item for item in x if item not in useless])

'''#Stemming
stemmer = PorterStemmer()
token = token.apply(lambda x:[stemmer.stem(i) for i in x])

for i in range(len(token)):
    token[i] = ' '.join(token[i])

data['tidy_tweet'] = token'''

'''#Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
token = token.apply(lambda x:[lemmatizer.lemmatize(i) for i in x])

for i in range(len(token)):
    token[i] = ' '.join(token[i])

data['tidy_tweet'] = token'''

#Visualization
#freq of words in whole data
all_words = ' '.join([text for text in data['tidy_tweet']])
all_words = all_words.split(' ')
#all_words = [x for x in all_words if x not in useless]
#all_words = [x for x in all_words if x not in stop_words]
counts = Counter(all_words)

#freq of words in happy senti
train_viz = data[:n_train]
train_viz['label'] = y_train

happy_words = ' '.join([text for text in train_viz['tidy_tweet'][train_viz['label']==0]])
happy_words = happy_words.split(' ')
counts2 = Counter(happy_words)

#freq of sad words
sad_words = ' '.join([text for text in train_viz['tidy_tweet'][train_viz['label']==1]])
sad_words = sad_words.split(' ')
counts3 = Counter(sad_words)

print(counts.most_common(10))
print(counts3.most_common(10))
print(counts2.most_common(10))

# importance in hashtag
def hashtag(x):
    hasht =[]
    for i in x:
        ht = re.findall(r"#(\w+)",i)
        hasht.append(ht)
    return hasht

hpositive = hashtag(train_viz['tidy_tweet'][train_viz['label']==0])
hnegative = hashtag(train_viz['tidy_tweet'][train_viz['label']==1])
#print(hpositive) - check what nested list looks like
#unnesing list
hpositive = sum(hpositive,[])
hnegative = sum(hnegative,[])
#print(hnegative)
#print(hpositive)

#Visualize
a = nltk.FreqDist(hpositive)
b = nltk.FreqDist(hnegative)
d_p = pd.DataFrame({'Hash_tag': list(a.keys()),'Count': list(a.values())})
d_n = pd.DataFrame({'Hash_tag': list(b.keys()),'Count': list(b.values())})

d_p = d_p.nlargest(columns='Count', n=10)
d_n = d_n.nlargest(columns='Count', n=10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d_n,x='Hash_tag',y='Count')
ax.set(ylabel = 'Count')
#plt.show()

#TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data['tidy_tweet'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
train_tfidf = tfidf[:n_train,:]
test_tfidf = tfidf[n_train:,:]
# splitting data into training and validation set
xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, y_train, random_state=42, test_size=0.3)

#Logistic regression
lreg = LogisticRegression()
lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
test_pred = lreg.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)

print(f1_score(yvalid,prediction_int))

#Submission
sub = pd.DataFrame()
sub['id'] = test_id
sub['label'] = test_pred_int
sub.to_csv('sub_lreg_tfidf_new_fet2.csv', index = False)
