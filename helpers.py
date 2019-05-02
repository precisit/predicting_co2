from datetime import datetime

import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor as dtr

from matplotlib.pyplot import *

def get_date():
	now = datetime.now()
	year = now.year
	month = now.month
	day =  year + month/12 + now.day/365 # decimal day
	return year, month, day

def get_day():
	year, month, day = get_date()
	return day

def get_today():
	year, month, day = get_date()
	# d = {'Year': [year], 'Month': [month], 'Date': [day]}
	# df = pd.DataFrame(data = d)
	return year, month, day

def get_mae(model, x, y): # val x y
	pred = model.predict(x)
	model_mae = mae(pred, y)
	print("rf:  {:,.6f}".format(model_mae))

def get_axis(x, y, feature):
	x = x[feature].sort_index(axis = 0)
	y = y.sort_index(axis = 0)
	return x, y

def get_plot(X, y, train_X, train_y, val_X, val_y, model_dt, model_rf, model_xgb, axis):
	xa, ya = get_axis(X, y, axis)
	xta, yta = get_axis(train_X, train_y, axis)
	xva, yva = get_axis(val_X, val_y, axis)

	xpa = get_day()

	# ya_dt = model_dt.predict(get_today())[0]
	# ya_rf = model_rf.predict(get_today())[0]
	# ya_xgb = model_xgb.predict(get_today())[0]

	subplot(3, 1, 1)
	xticks([]), yticks([])
	title('All data')
	plot(xa, ya) #, xpa, ya_dt, '*', xpa, ya_rf, '*',  xpa, ya_xgb, '*')

	subplot(3, 1, 2)
	xticks([]), yticks([])
	title('Training data')
	plot(xta, yta, 'r-')

	subplot(3, 1, 3)
	xticks([]), yticks([])
	title('Validation data')
	plot(xva, yva, 'g-')

	show()

'''
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = dtr(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    error = mae(val_y, preds_val)
    return(error)

# compare MAE with differing values of max leaf nodes
max_leaf_nodes = [5, 50, 500, 5000]
for i in max_leaf_nodes:
    my_mae = get_mae(i, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t MAE:  %f" %(i, my_mae))
'''