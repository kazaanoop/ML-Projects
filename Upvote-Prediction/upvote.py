import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.jpl_units import deg
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import boxcox
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
import seaborn as sns
#from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLars
from sklearn.metrics import mean_squared_error


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#dropping ID columns
train_id = train['ID']
test_id = test['ID']
train = train.drop(columns = 'ID' , axis =1)
test = test.drop(columns='ID', axis=1)

#Dropping rows with views greater than 3000000
train = train.drop(train[train.Views>3000000].index)

#print(train)

#Splitting the target value
n_train = train.shape[0]
n_test = test.shape[0]
y_train = train['Upvotes']
train = train.drop(columns = 'Upvotes' ,axis =1 )
data = pd.concat((train,test)).reset_index(drop=True)
data = data.drop(columns='Username' ,axis =1 )
'''
#adding 1 to all rows in order to apply log transformation
data['Answers'] = data['Answers'].apply(lambda x: x+1)
data['Reputation'] = data['Reputation'].apply(lambda x: x+1)

#Feature extraction
data['Views'] = pd.cut(data.Views, bins=[0, 10000, 50000, 200000000], labels=[1,2,3])
data['Reputation'] = pd.cut(data.Reputation, bins=[0, 1000, 10000, 800000000], labels=[1,2,3])
data['Answers'] = pd.cut(data.Answers, bins=[0, 5, 20, 10000], labels=[1,2,3])

'''
#Feature extraction
#data['View_level'] = pd.cut(data.Views, bins=[0, 10000, 50000, 200000000], labels=[1,2,3])
#data['Reputation_level'] = pd.cut(data.Reputation, bins=[0, 1000, 10000, 800000000], labels=[1,2,3])
#data['Answers_level'] = pd.cut(data.Answers, bins=[0, 7, 20, 10000], labels=[1,2,3])
data['Answers_level'] = pd.cut(data.Answers, bins=[-10, 7, 10000], labels=[0,1])

#sns.distplot(train['Reputation'])
#plt.show()

#checking correlation
#train.corr()
#plt.show()

#Removing the tail values to check the distribution
#data = data[data['Views']<10000]

#Checking distribution of target data
#data.hist(column = 'Views' ,bins= 10)
#data.hist(column = 'Answers' ,bins= 10)
#data.hist(column = 'Reputation' ,bins= 10)
#plt.show()

#print(data.describe())

#Allplying transormation(Log) as data is not normally distributed
#data['Views'] = boxcox(data['Views'],0)
#data['Answers'] = boxcox(data['Answers'],0)
#data['Reputation'] = boxcox(data['Reputation'],0)
#qqplot(data['Views'], line='r')
#plt.show()

#data.hist(column = 'Views' ,bins= 10)
#data.hist(column = 'Answers' ,bins= 10)
#data.hist(column = 'Reputation' ,bins= 10)
#plt.show()

#sns.distplot(data['Answers'])
#plt.show()

#Label encoding
lbl = LabelEncoder()
lbl.fit(list(data['Tag'].values))
data['Tag'] = lbl.transform(list(data['Tag'].values))

#Checking symmetry of numeric features
Num_fet = data.dtypes[data.dtypes != "object"].index
skew_fet = data[Num_fet].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
print(skew_fet)

#test and train
train = data[:n_train]
test = data[n_train:]

print(train.shape)
print(test.shape)

print(data)

#visulaization
#sns.pairplot(train2)
#plt.show()

'''
#Polynomial regression
X_train, X_test, y_train1, y_test = train_test_split(train, y_train, test_size=0.3, random_state=101)

degree = 2
for i in range(2,5):
    poly_features = PolynomialFeatures(degree=i)
    X_train_poly = poly_features.fit_transform(X_train)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train1)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train1, y_train_predicted))
    #r2_train = r2_score(Y_train, y_train_predicted)
    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))

    print("degree",i)
    print("rmse-train",rmse_train)
    print("rmse-test",rmse_test)
'''

#predicting on actual test
for i in range(2,3):
    poly_features = PolynomialFeatures(degree=i)
    X_train_poly = poly_features.fit_transform(train)
    poly_model = LinearRegression()
    sc = StandardScaler()
    X_train_poly = sc.fit_transform(X_train_poly)
    #poly_model = LassoLars(alpha=0.021,max_iter=150)
    poly_model.fit(X_train_poly, y_train)
    X_test_poly = poly_features.fit_transform(test)
    X_test_poly = sc.fit_transform(X_test_poly)

        # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    print(X_train_poly.shape)
    print(X_test_poly.shape)
        # predicting on test data-set
    y_test_predict = poly_model.predict(X_test_poly)

        # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    print("degree :", i)
    print(rmse_train)

#Exporting results
sub = pd.DataFrame()
sub['ID'] = test_id
sub['Upvotes'] = abs(y_test_predict)
sub.to_csv('poly_reg_degree_2_abs_no_normal_scaler_1fet.csv', index = False)


'''
#linear regression - failed because 1.no linear relationship with upvotes 2.many outliers 3.multi Multicollinearity 
#X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(train,y_train)

#predictions = lm.predict(X_test)
predictions2 = lm.predict(test)
#Exporting results
sub = pd.DataFrame()
sub['ID'] = test_id
sub['Upvotes'] = predictions2
sub.to_csv('linear_reg.csv', index = False)

#plt.scatter(y_test,predictions)
#plt.show()


#Modelling
# importing libraries
from sklearn.linear_model import ElasticNet,Lasso,BayesianRidge,LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
#import xgboost as xgb
#import lightgbm as lgb

#Cross validation - using cross_val_score
kfolds = 100

def rmsle_cv(model):
    kf = KFold(kfolds,shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,train.values,y_train,scoring="neg_mean_squared_error",cv=kf))
    return rmse

#Lasso
lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))

#Enet
enet = make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=0.9,random_state=3))

alphas = [1e-4, 5e-4, 1e-3, 5e-3]
cv_enet = [rmsle_cv(ElasticNet(alpha = alpha, max_iter=50000)) for alpha in alphas]
#print(cv_enet)

model_enet = ElasticNet(alpha=1e-4, max_iter=50000).fit(train,y_train)

#Kernel Ridge

kr = make_pipeline(RobustScaler(),KernelRidge(alpha=0.6,kernel='polynomial',degree=2, coef0=2.5))

print(rmsle_cv(lasso).mean())
print(rmsle_cv(enet).mean())
#print(rmsle_cv(kr).mean())

#enet.fit(train,y_train)

#Exporting results
sub = pd.DataFrame()
sub['ID'] = test_id
sub['Upvotes'] = model_enet.predict(test)
sub.to_csv('enet.csv', index = False)

'''