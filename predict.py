import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
import numpy as np

from helpers import get_day, get_today, get_plot, get_mae
from data import get_data

# models
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr
from xgboost import XGBRegressor as xgb

import warnings
warnings.filterwarnings("ignore")

# variables
PATH = 'mod.csv'
FEATURES = ['Year', 'Month', 'Date']
AXIS = 'Date'

# FIXME para med co2-data, befolkningsöknning, ökad tillväxt, mer?
print('\n')
print('--- start ---')
print('\n')

# get data
data = get_data(PATH)
y = data.CO2 # set prediction metric
X = data[FEATURES]

# split to validation and training data
train_X, val_X, train_y, val_y = tts(X, y, random_state = 1)

print('validation MAEs')

# decision tree
model_dt = dtr(random_state = 1)
model_dt.fit(train_X, train_y)
get_mae(model_dt, val_X, val_y)

# random forest
model_rf = rfr(random_state = 1)
model_rf.fit(train_X, train_y)
get_mae(model_rf, val_X, val_y)

# xgboost
model_xgb = xgb(random_state = 1, n_estimators = 10000, learning_rate = 0.01)
model_xgb.fit(train_X, train_y, early_stopping_rounds = 10, eval_set = [(val_X, val_y)], verbose = False)
get_mae(model_xgb, val_X, val_y)

# make predictions for today
user_X = get_today()
print('\n')
print('predicted ppm CO2 in atmosphere today')
print('dt:  ', model_dt.predict(user_X)[0])
print('rf:  ', model_rf.predict(user_X)[0])
print('xgb: ', model_xgb.predict(user_X)[0])

# plot
get_plot(X, y, train_X, train_y, val_X, val_y, model_dt, model_rf, model_xgb, AXIS)

print('\n')
print('done!')
print('\n')
