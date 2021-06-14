
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from matplotlib import pyplot as plt
import seaborn as sea
import squarify   
import lightgbm as lgb
import gc


aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
priors = pd.read_csv('order_products__prior.csv')
train = pd.read_csv('order_products__train.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')

print("aisles", aisles.shape, aisles.columns)
print("departments:", departments.shape, departments.columns)
print("priors:", priors.shape, priors.columns)
print("train:", train.shape, train.columns)
print("orders:", orders.shape, orders.columns)
print("products:", products.shape, products.columns)

best50 = priors['product_id'].value_counts()[0:50].to_frame().reset_index()
# print(best50)
# print((products[products['product_id']==472565]['product_name'].iloc[0]))
name = []
for id in best50['index']:
    name.append(products[products['product_id']==id]['product_name'].iloc[0])
# print(name)
# sea.barplot(best50['product_id'][0:7],name[0:7])
sells = pd.DataFrame({
    'Name': np.array(name)[0:8],
    "Volume": best50['product_id'][0:8]
})
plt.figure(figsize=(16,8))
pic = sea.barplot(x='Name', y='Volume', data=sells)
pic.set_xticklabels(pic.get_xticklabels(), rotation=90)

"""# 賣最好的產品，列出販賣次數前幾高的產品"""


"""# 統計training data中商品連續被再次購買的次數(連續兩次購買相同物品，reordered即會被設為1)"""

plt.figure(figsize=(10,5))
reordered = pd.DataFrame({
    'Reorder':['1','0'],
    'Times':train['reordered'].value_counts()
})
print(reordered)
sea.barplot(x='Reorder',y='Times',data=reordered)

"""# 商品被再次購買的比率(每次購買中)

# 統計Order_dow(一週中的哪一天購買)的數量，因資料沒有特別註明數字分別代表禮拜幾，只能找出第幾天最常購買
"""

Order_dow = orders['order_dow'].value_counts().to_frame().reset_index()
plt.figure(figsize=(10,5))
sea.barplot(x='index',y='order_dow',data=Order_dow)

"""# 統計兩次購買間間隔幾天
可以看出一週內再次購買的機率相當高，而最後一天30天飆高可能是超過30天購買第二次都歸類在30天
"""

twodays = orders['days_since_prior_order'].value_counts().to_frame().reset_index()
# print(twodays)
plt.figure(figsize=(15,5))
sea.barplot(x='index',y='days_since_prior_order',data=twodays)

"""# 統計每次購買的時間點(幾點鐘)
可以看出白天(約7~19)購買的機率最高
"""

hours = orders['order_hour_of_day'].value_counts().to_frame().reset_index()
plt.figure(figsize=(15,5))
sea.barplot(x='index',y='order_hour_of_day',data=hours)

"""# 因為資料量過大，記憶體有限，將資料型態由int64轉成int8和int32(根據數據大小)。
## Ex:orders中order_dow為0~6，故轉成int8
## 而orders中order_id最大為3421083，故轉成int32

### 印出每筆column最大範圍，決定更改型態
"""

print('priors:order_id', max(priors.order_id))
print('priors:product_id', max(priors.product_id))
print('priors:add_to_cart_order', max(priors.add_to_cart_order))
print('priors:reordered', max(priors.reordered))
print('orders:user_id', max(orders.user_id))
print('orders:order_number', max(orders.order_number))
print('orders:order_hour_of_day', max(orders.order_hour_of_day))
print('orders:days_since_prior_order', max(orders.days_since_prior_order[1:]))
print('products:aisle_id', max(products.aisle_id))
print('products:department_id', max(products.department_id))

orders.order_dow = orders.order_dow.astype(np.int8)
orders.order_hour_of_day = orders.order_hour_of_day.astype(np.int8)
orders.order_number = orders.order_number.astype(np.int16)
orders.order_id = orders.order_id.astype(np.int32)
orders.user_id = orders.user_id.astype(np.int32)
orders.days_since_prior_order = orders.days_since_prior_order.astype(np.float32)

products.drop(['product_name'], axis=1, inplace=True)
products.aisle_id = products.aisle_id.astype(np.int8)
products.department_id = products.department_id.astype(np.int8)
products.product_id = products.product_id.astype(np.int32)

train.order_id = train.order_id.astype(np.int32)
train.reordered = train.reordered.astype(np.int8)
train.add_to_cart_order = train.add_to_cart_order.astype(np.int16)

priors.order_id = priors.order_id.astype(np.int32)
priors.add_to_cart_order = priors.add_to_cart_order.astype(np.int16)
priors.reordered = priors.reordered.astype(np.int8)
priors.product_id = priors.product_id.astype(np.int32)

"""# 計算先前某項產品重複購買的頻率(rate = reorders/orders)"""

prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.float32)
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products.join(prods, on='product_id') #依照product_id來排序 並把prods加進products
products.set_index('product_id', drop=False, inplace=True)
del prods

print('add order info to priors')
orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_new')
priors.drop('order_id_new', inplace=True, axis=1)

"""# 創建一個新的DataFrame:user紀錄每個用戶以下資訊
1. Total_item:總共買了幾樣產品
2. all_products_id:全部買的產品的product_id
3. total_different_item:總共買過哪些不同的產品
4. average_days:平均幾天買一次
5. average_times:平均在一天的何時購買
6. number_orders:購買的次數
7. average_buy:平均一次購買幾樣產品
"""

usr = pd.DataFrame()
usr['average_days'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['average_times'] = orders.groupby('user_id')['order_hour_of_day'].mean().astype(np.float32)
usr['most_dow'] =  orders.groupby('user_id')['order_dow'].agg(lambda x:x.value_counts().index[0]).astype(np.int8) # 利用value_counts()找出出現最多次的dow
usr['number_orders'] = orders.groupby('user_id').size().astype(np.int16)

users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16) # 計算總共買了多少數量的物品
users['all_products_id'] = priors.groupby('user_id')['product_id'].apply(set) # 計算買了哪些物品
users['total_different_item'] = (users.all_products_id.map(len)).astype(np.int16) #計算不同物品的數量

users = users.join(usr)
del usr
users['average_buy'] = (users.total_items / users.number_orders).astype(np.float32)
gc.collect()
print('user f', users.shape)


priors['user_product'] = priors.product_id + priors.user_id * 100000

d= dict()
for row in priors.itertuples():
    z = row.user_product
    if z not in d:
        d[z] = (1, (row.order_number, row.order_id), row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1, max(d[z][1], (row.order_number, row.order_id)), d[z][2] + row.add_to_cart_order)
d = pd.DataFrame.from_dict(d, orient='index')
d.columns = ['number_orders', 'last_order_id', 'sum_pos_in_cart']
d.number_orders = d.number_orders.astype(np.int16)
d.last_order_id = d.last_order_id.map(lambda x: x[1]).astype(np.int32)
d.sum_pos_in_cart = d.sum_pos_in_cart.astype(np.int16)

user_product = d
print('user X product f', len(user_product))

del priors


"""# 切割train/test data，透過orders的eval_set column來區分"""

test_orders = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']

train.set_index(['order_id', 'product_id'], inplace=True, drop=False)


"""# 模型"""

def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i=0
    for row in selected_orders.itertuples():
        i+=1
        if i%10000 == 0: print('order row',i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products_id[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]
        
    df = pd.DataFrame({'order_id':order_list, 'product_id':product_list})
    df.order_id = df.order_id.astype(np.int32)
    df.product_id = df.product_id.astype(np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    
    df['user_id'] = df.order_id.map(orders.user_id).astype(np.int32)
    df['user_total_orders'] = df.user_id.map(users.number_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_different_item)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days)
    df['user_average_basket'] =  df.user_id.map(users.average_buy)
    df['user_average_times'] = df.user_id.map(users.average_times) #
    df['user_most_dow'] = df.user_id.map(users.most_dow)
    
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    
    df['aisle_id'] = df.product_id.map(products.aisle_id).astype(np.int8)
    df['department_id'] = df.product_id.map(products.department_id).astype(np.int8)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.float32)
    df['product_reorders'] = df.product_id.map(products.reorders).astype(np.float32)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    df['z'] = df.user_id * 100000 + df.product_id
    df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(user_product.number_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(user_product.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(user_product.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - \
                  df.UP_last_order_id.map(orders.order_hour_of_day)).map(lambda x: min(x, 24-x)).astype(np.int8)

    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)

    gc.collect()
    return (df, labels)

df_train, labels = features(train_orders, labels_given=True)

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket', 'user_average_times', 'user_most_dow',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last']


print('formating for lgb')
d_train = lgb.Dataset(df_train[f_to_use],
                      label=labels,
                      categorical_feature=['aisle_id', 'department_id'])
del df_train
gc.collect()

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 98

bst = lgb.train(params, d_train, ROUNDS)
lgb.plot_importance(bst, figsize=(9,20))
del d_train
gc.collect()

df_test, _ = features(test_orders)
preds = bst.predict(df_test[f_to_use])

df_test['pred'] = preds

TRESHOLD = 0.22  

d = dict()
for row in df_test.itertuples():
    if row.pred > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test_orders.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('submission.csv', index=False)