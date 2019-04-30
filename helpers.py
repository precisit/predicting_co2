from datetime import datetime

import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor as dtr

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = dtr(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    error = mae(val_y, preds_val)
    return(error)

def get_today():
	now = datetime.now()
	year = now.year
	month = now.month
	day = now.day
	day_decimal =  year + month/12 + day/365
	d = {'Year': [year], 'Month': [month], 'Date': [day_decimal]}
	df = pd.DataFrame(data = d)
	return(df)

'''
# compare MAE with differing values of max leaf nodes
max_leaf_nodes = [5, 50, 500, 5000]
for i in max_leaf_nodes:
    my_mae = get_mae(i, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t MAE:  %f" %(i, my_mae))
'''