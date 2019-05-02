import pandas as pd
import numpy as np
from matplotlib.pyplot import *

PATH_CO2 = 'mod.csv'
PATH_OCEAN = 'hav_2017.xlsx'
PATH_RAIN = 'nbd_ar_tom_2017.xls'
PATH_TEMP = 'temp_ar_tom_2018.xls'


def get_data(path):
	# load and prep co2 data
	d = pd.read_csv(PATH_CO2)

	# load swe ocean rain and temp data
	d_ocean = pd.read_excel(PATH_OCEAN)
	d_rain = pd.read_excel(PATH_RAIN)
	d_temp = pd.read_excel(PATH_TEMP)
	d_ocean = d_ocean[d_ocean['year'] >= 1958]
	d_rain = d_rain[d_rain['year'] >= 1958]
	d_temp = d_temp[d_temp['year'] >= 1958]

	#d['OM'] = d_ocean['ocean_mean']

	# kommer behöva 
	for i, row in d.iterrows():
		for j, row in d_ocean.iterrows():
			if d.at[i, 'Year'] == d_ocean.at[j, 'year']:
				d.at[i, 'OM'] = d_ocean.at[j, 'ocean_mean']
				d.at[i, 'OD'] = d_ocean.at[j, 'ocean_delta']
		for j, row in d_rain.iterrows():
			if d.at[i, 'Year'] == d_rain.at[j, 'year']:
				d.at[i, 'R'] = d_rain.at[j, 'rain']
				d.at[i, 'RM'] = d_rain.at[j, 'rain_mean']
		for j, row in d_temp.iterrows():
			if d.at[i, 'Year'] == d_temp.at[j, 'year']:
				d.at[i, 'T'] = d_temp.at[j, 'temp']
				d.at[i, 'TS'] = d_temp.at[j, 'temp_std']
				d.at[i, 'TD'] = d_temp.at[j, 'temp_dmean']

	d = d.dropna()
	return d