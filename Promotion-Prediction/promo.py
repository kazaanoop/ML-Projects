import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import boxcox
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
#import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


#dropping ID columns
train_id = train['employee_id']
test_id = test['employee_id']
train = train.drop(columns = 'employee_id' , axis =1)
test = test.drop(columns='employee_id', axis=1)

#visualization
plot_data1 = train.groupby('department')['is_promoted'].count().values
plot_data2 = train.groupby('department')['is_promoted'].count().index
#plt.bar(plot_data2,plot_data1)

plot_data1 = train.groupby('region')['is_promoted'].count().values
plot_data2 = train.groupby('region')['is_promoted'].count().index
#plt.bar(plot_data2,plot_data1)
#plt.show()
# checking count of target values
pd.value_counts(train['is_promoted']).plot.bar()
#plt.show() - immbalance data
pd.value_counts(train['education']).plot.bar()

#print(train['department'][train['is_promoted'] == 0].count())

#print(train.groupby('department')['department'])
#print("hello   ")
#print(train.groupby('region')['is_promoted'].count())

#Splitting the target value
n_train = train.shape[0]
n_test = test.shape[0]
y_train = train['is_promoted']
train = train.drop(columns = 'is_promoted' ,axis =1 )
data = pd.concat((train,test)).reset_index(drop=True)

#Missing values
data['education'] = data['education'].fillna(data['education'].mode()[0])
data['previous_year_rating'] = data['previous_year_rating'].fillna(data['previous_year_rating'].mode()[0])

# Feature engineering col == 'HR' || col == 'Legal' || col == 'R&D'
#for i,col in data['department']:
#    if col == 'Finance':
 #       data['department'][i]= 'Other' 

# Failed attempt (trying to replace only some columns as 'other')
#data['department'] = data['department'].map({'Finance' : 'Other','HR' : 'Other','Legal' : 'Other','R&D' : 'Other'})
data['department'] = data['department'].replace(['Finance','HR','Legal','R&D'],['Other','Other','Other','Other',])
#data['region'] = data['region'].replace(['region_34','region_3','region_33','region_18'],['Other','Other','Other','Other',])
#print(data['department'])

data['age'] = pd.cut(data.age, bins=[19, 30, 40, 60], labels=["set1", "set2", "set3"])
data['avg_training_score'] = pd.cut(data.avg_training_score, bins=[38, 60, 70, 80, 100], labels=["part1", "part2", "part3", "part4"])
data['length_of_service'] = pd.cut(data.length_of_service, bins=[0, 3, 6, 30], labels=["fresher", "mid_exp", "high_exp"])

#Checking for correlation
data_corr = data.corr().abs()
#print(data_corr)

#Allplying transormation(Log) as data is not normally distributed
data['no_of_trainings'] = boxcox(data['no_of_trainings'],0)
data['previous_year_rating'] = boxcox(data['previous_year_rating'],0)

#feature eng and new features
data['education'].replace("Master's & above",3,inplace = True)
data['education'].replace("Bachelor's",2,inplace = True)
data['education'].replace("Below Secondary",1,inplace = True)
data['sum_metric'] = data['awards_won?']+data['KPIs_met >80%']+data['previous_year_rating']
data['tot_score'] = data['no_of_trainings']*data['avg_training_score']

#Label encoding
cols = ('department','region','gender','recruitment_channel','age')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(data[c].values))
    data[c] = lbl.transform(list(data[c].values))

# Dummies
#data = pd.get_dummies(data)

#test and train
train = data[:n_train]
test = data[n_train:]

#splitting data to validation data.
X_train,X_test,y_train,y_test = train_test_split(train,y_train,test_size=0.2,random_state=0)

#Applying Oversampling - SMOTE
'''kfold = StratifiedKFold(n_splits=10)'''

#applying model Logistic reg
'''logreg = LogisticRegression()
logreg.fit(train,y_train)
y_pred = logreg.predict(test)
accu_LR = round(logreg.score(train,y_train)*100,2)
print(accu_LR)'''

'''sm = SMOTE(random_state=2)
X_train_os,y_train_os = sm.fit_resample(train,y_train.ravel())'''

#Applying PCA
pca = PCA(.95)
prinComp_train = pca.fit(train)
prinComp_test = pca.fit(test)


#Decision tree
rf = RandomForestClassifier(n_estimators=100)
rf.fit(prinComp_train,y_train)
y_pred2 = rf.predict(prinComp_test)
accu_rf = round(rf.score(prinComp_train,y_train)*100,2)
print(accu_rf)
'''
#Applying Lightgbm
categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col]
train_data = lightgbm.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
test_data = lightgbm.Dataset(X_test, label=y_test)

#Training Model
parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 10,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lightgbm.train(parameters,train_data,valid_sets=test_data,num_boost_round=5000,early_stopping_rounds=100)

y_pred_light = model.predict(test)
'''
#Ada boost
'''model = GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(train,y_train)
y_pred3 = model.predict(test)
accu_GB = round(model.score(train,y_train)*100,2)
print(accu_GB)'''
'''
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],"base_estimator__splitter" :["best", "random"],
                  "algorithm":["SAMME","SAMME.R"],"n_estimators":[1,2],"learning_rate": [0.0001,0.001,0.01,0.1,0.2,0.3,1.5]}
gsadaDTC = GridSearchCV(adaDTC, param_grid =ada_param_grid, cv = kfold ,scoring="accuracy",n_jobs=4,verbose=1)
gsadaDTC.fit(train,y_train)
ada_best=gsadaDTC.best_estimator_

RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[50,100],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(train,y_train)

RFC_best = gsRFC.best_estimator_
'''
# Best score
#gsRFC.best_score_


'''SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [0.1, 1],
                  'C': [1, 10, 50]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)

gsSVMC.fit(train,y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_'''

'''
# Voting classifier
vote = VotingClassifier(estimators=[('rfc',RFC_best),('adac',ada_best)],voting='soft',n_jobs=4)
vote = vote.fit(train,y_train)

sub = pd.DataFrame()
sub['employee_id'] = test_id
sub['is_promoted'] = vote.predict(test)
sub.to_csv('submission_vote.csv', index = False)'''

#sub = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived":y_pred2})
'''sub = pd.DataFrame()
sub['employee_id'] = test_id
sub['is_promoted'] = y_pred2
sub.to_csv('submission.csv', index = False)

sub2 = pd.DataFrame()
sub2['employee_id'] = test_id
sub2['is_promoted'] = y_pred
sub2.to_csv('submission2.csv', index = False)
'''

sub = pd.DataFrame()
sub['employee_id'] = test_id
sub['is_promoted'] = y_pred2
sub.to_csv('submission_PCA.csv', index = False)

'''
sub = pd.DataFrame()
sub['employee_id'] = test_id
sub['is_promoted'] = y_pred_light
sub.to_csv('submission_light.csv', index = False)'''

