import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#Model imports
'''from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier'''

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#removing Id
train_df = train_df.drop(columns = 'PassengerId', axis =1)
test_id = test_df['PassengerId']
test_df = test_df.drop(columns = 'PassengerId', axis =1)

#Checking outliers
#plt.scatter(x=train_df['Fare'], y=train_df['Pclass'])
#plt.show()

n_train = train_df.shape[0]
n_test = test_df.shape[0]
y_train = train_df['Survived']
train_df = train_df.drop(columns = 'Survived' ,axis =1 )
data = pd.concat((train_df,test_df)).reset_index(drop=True)

#Missing values
#print(data[data.columns[:]].isnull().sum()*100/len(data))
#Removing cabin column as it has 77% missing values
data = data.drop(columns='Cabin',axis=1)
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

#filled missing values with mean(group by Pclass)
data['Fare'] = data['Fare'].replace(0,np.NaN)
values = data.groupby('Pclass')['Fare'].transform('mean')
data['Fare'] = np.where(data['Fare'].notnull(),data['Fare'],values).astype(int)
#data = data.replace({'Fare':0},)
#print(values)

#extracting master,mr,mrs,miss from Name
data_title = [i.split(',')[1].split('.')[0].strip() for i in data['Name']]
data['Title'] = pd.Series(data_title)
data = data.drop(columns = 'Name', axis =1)

#filled missing values with mean(group by Title)
values1 = data.groupby('Title')['Age'].transform('mean')
data['Age'] = np.where(data['Age'].notnull(),data['Age'],values1).astype(int)

#Checking for correlation
data_corr = data.corr().abs()
#print(data_corr)

#Checking distribution of Fare
#qqplot(data['Fare'], line='r')
#plt.show()

#Allplying transormation(Log) as data is not normally distributed
data['Fare'] = boxcox(data['Fare'],0)

data = data.drop(columns = 'Ticket',axis =1)
data['Age'] = pd.cut(data.Age, bins=[0, 13, 22, 40, 60, 100], labels=["infant", "teen", "young", "men", "old"])
data['Pclass'] = data['Pclass'].astype(str)
data['SibSp'] = data['SibSp'].astype(str)
data['Parch'] = data['Parch'].astype(str)

#Label encoding
cols = ('Pclass','SibSp','Parch','Age','Sex','Embarked','Title')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(data[c].values))
    data[c] = lbl.transform(list(data[c].values))

#test and train
train = data[:n_train]
test = data[n_train:]

#applying model Logistic reg
logreg = LogisticRegression()
logreg.fit(train,y_train)
y_pred = logreg.predict(test)
accu = round(logreg.score(train,y_train)*100,2)

#Decision tree
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train,y_train)
y_pred2 = rf.predict(test)
accu_rf = round(rf.score(train,y_train)*100,2)
print(accu_rf)

#sub = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived":y_pred2})
sub = pd.DataFrame()
sub['PassengerId'] = test_id
sub['Survived'] = y_pred2
sub.to_csv('submission.csv', index = False)
#print(data[data.columns[:]].isnull().sum()*100/len(data))
