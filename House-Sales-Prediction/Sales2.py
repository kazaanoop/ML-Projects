import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
item_cat = pd.read_csv('item_categories.csv')
items = pd.read_csv('items.csv')
sam_sub = pd.read_csv('sample_submission.csv')
shops = pd.read_csv('shops.csv')

'''data = pd.merge(train,items)

y = data['item_cnt_day']
data = data.drop(columns = ['item_cnt_day'] ,axis =1)

data2 = pd.merge(test,items)

full_data = pd.concat((data,data2)).reset_index(drop=True)
print(full_data.head())'''

train.date = train.date.apply(lambda x:datetime.datetime.strptime(x,'%d.%m.%Y'))
monthly_sales = train.groupby(["date_block_num","shop_id","item_id"])["date","item_price","item_cnt_day"].agg({'date':['min','max'],'item_price':'mean','item_cnt_day':'sum'})

item_count = items.groupby(["item_category_id"]).count()

#plt.bar(item_count.item_category_id,item_count.item_id)
#plt.xlabel("category")
#plt.ylabel("item count")
sal_mon = train.groupby(['date_block_num'])['item_cnt_day'].sum()

#plt.plot(sal_mon)
#plt.show()

# checking trend and seasonality
result = sm.tsa.seasonal_decompose(sal_mon.values , model = 'multiplicative', freq =1)
result.plot()
plt.show()

#print(item_count)