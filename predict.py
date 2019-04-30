import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr
from xgboost import XGBRegressor as xgb
import matplotlib.pyplot as plt
import numpy as np

from helpers import get_today

import warnings
warnings.filterwarnings("ignore")
# FIXME här ska man få ladda in ett dataset, bestämma target, bestämma features
# FIXME para med co2-data, befolkningsöknning, ökad tillväxt, mer?

print('\n')
print('--- start ---')
print('\n')

# perp data
path = 'mod.csv'
data = pd.read_csv(path)
data = data.dropna()

y = data.CO2
features = ['Year', 'Month', 'Date']
X = data[features]

# split to validation and training data
train_X, val_X, train_y, val_y = tts(X, y, random_state = 1)

print('validation MAEs')

# decision tree
model_dt = dtr(random_state = 1)
model_dt.fit(train_X, train_y)
pred_dt = model_dt.predict(val_X)
mae_dt = mae(pred_dt, val_y)
print("dt:  {:,.6f}".format(mae_dt))

# random forest
model_rf = rfr(random_state = 1)
model_rf.fit(train_X, train_y)
pred_rf = model_rf.predict(val_X)
mae_rf = mae(pred_rf, val_y)
print("rf:  {:,.6f}".format(mae_rf))

# xgboost

model_xgb = xgb(random_state = 1, n_estimators = 10000, learning_rate = 0.01)
model_xgb.fit(train_X, train_y, early_stopping_rounds = 10, eval_set = [(val_X, val_y)], verbose = False)
pred_xgb = model_xgb.predict(val_X)
mae_xgb = mae(pred_xgb, val_y)
print("xgb: {:,.6f}".format(mae_xgb))

# let user make predictions
# user_try = input('do you wanna make a prediction for today? [y/n] ')
# while user_try.lower() != 'n' or user_try.lower() != 'no':
#	if user_try.lower() == 'y' or user_try.lower() == 'yes':
user_X = get_today()
print('\n')
print('predicted ppm CO2 in atmosphere today')
print('dt:  ', model_dt.predict(user_X)[0])
print('rf:  ', model_rf.predict(user_X)[0])
print('xgb: ', model_xgb.predict(user_X)[0])
#		break
#	else:
#		user_try = input('weird response - what did you mean? [y/n] ')
#		if user_try.lower() != 'n' or user_try.lower() != 'no':
#			break


print('\n')
print('done!')
print('\n')
