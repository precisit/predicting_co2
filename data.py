import pandas as pd
import numpy as np
from matplotlib.pyplot import *

PATH_CO2 = 'mod.csv'

PATH_OCEAN = 'hav_2017.xlsx'

PATH_RAIN_FALL = 'nbd_host_tom_2017.xls'
PATH_RAIN_WIN = 'nbd_vin_tom_2017.xls'
PATH_RAIN_SPR = 'nbd_var_tom_2017.xls'
PATH_RAIN_SUM = 'nbd_som_tom_2017.xls'

PATH_TEMP_FALL = 'temp_hos_tom_2018.xls'
PATH_TEMP_WIN = 'temp_vin_tom_2018.xls'
PATH_TEMP_SPR = 'temp_var_tom_2018.xls'
PATH_TEMP_SUM = 'temp_som_tom_2018.xls'

# load and prep co2 data
d = pd.read_csv(PATH_CO2)
d = d.dropna()

# load ocean data swe
d_ocean = pd.read_excel(PATH_OCEAN)
d_ocean = d_ocean.dropna()
d_ocean = d_ocean[d_ocean['year'] >= 1958]

#d['OM'] = d_ocean['ocean_mean']

#for i in range(min(d_ocean['year']), max(d_ocean['year'])): # gå igenom alla åren
	# jag vill få index av d och ge dem nya värden
	
	#print(d.loc[d['Year'] == i]) #['OM' = range(12)]#, 'ocean_mean'])

for i, row in d.iterrows():
	if d.at[i, 'Year'] == 2017:
		d.at[i, 'OM'] = 3


d.loc[733, d.columns.get_loc('OM')] = 1
print(d)
#d_co2['RAIN'] = np.nan
#d_co2['TEMP'] = np.nan

#plot(range(len(d_ocean)), d_ocean['ocean_delta'])
#show()


#print(d_ocean.head())
#print(d.head())

#print(d_ocean.loc[d_ocean['år'] == 1958])

# print(d_co2.describe())

# load rain data swe

# load temp data swe

# load and prep xls files

def get_data(path):
	data = pd.read_csv(path)
	data = data.dropna()
	return data