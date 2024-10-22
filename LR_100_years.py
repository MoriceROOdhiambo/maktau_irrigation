"""Run aquacrop in Maktau


"""

import os
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'

import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import seaborn as sns

import atm

import warnings

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

supress_warnings = True

warnings.filterwarnings('ignore') 

from SeqMetrics import nse, rmse, RegressionMetrics, plot_metrics, agreement_index, r2, mae, nrmse, pbias

from scipy import stats

#################################################
# Read Maktau AWS data and prepare model inputs #
#################################################

met = pd.read_csv('MAWG_100years_input_2014_to_2023_formatted.dat')

met = met.set_index(pd.to_datetime(met['Date']))

dd = met

dd['Date'] = dd.index.values
dd['Precipitation'] = dd.rain.values
dd['ReferenceET'] = dd.pet.values
dd['MinTemp'] = dd.tmin.values
dd['MaxTemp'] = dd.tmax.values

# Model input

wdf = dd[['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET' , 'Date']]

#######################
# Aquacrop simulation #
#######################

sim_start = '2001/03/17'

sim_end = '2100/08/13'

custom = Soil('custom', dz=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

custom.add_layer(thickness=0.1, thWP=0.06, thFC=0.17, thS=0.45, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.1, thWP=0.08, thFC=0.20, thS=0.42, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.1, thWP=0.08, thFC=0.22, thS=0.42, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.2, thWP=0.10, thFC=0.23, thS=0.42, Ksat=1000, penetrability=100)

custom.add_layer(thickness=0.5, thWP=0.10, thFC=0.23, thS=0.42, Ksat=1000, penetrability=100)

# Initial water content

#initWC = InitialWaterContent(wc_type='Num', method='Layer', depth_layer=[1,2,3,4,5], value=[0.1,0.1,0.1,0.1,0.1])

initWC = InitialWaterContent(wc_type='Prop', method='Layer', depth_layer=[1,2,3,4,5], value=['FC','FC','FC','FC','FC'])

#initWC = InitialWaterContent(wc_type='Prop', method='Layer', depth_layer=[1,2,3,4,5], value=['WP','WP','WP','WP','WP'])

# Crop. Maktau Maize is DH02
               
maizeDH02=Crop('Maize', planting_date='03/17', PlantMethod=1, CalenderType=1, CGC=0.163, CDC=0.117,
               Emergence=6, Flowering=13, HIstart=66, YldForm=61, MaxRooting=108, Senescence=107, Maturity=132, HI0=0.48,
               Zmin=0.3, Zmax=1.0, PlantPop=37600, WP=33.7, CCx=0.88,                
               fshape_w1=2.9,fshape_w2=6.0, fshape_w3=2.7)

# Irrigation

rainfed = IrrigationManagement(irrigation_method=0)

# Output labels

outputs_water_flux=[]
outputs_water_storage=[]
outputs_crop_growth=[]
outputs_final_stats=[]

model = AquaCropModel(sim_start,sim_end,wdf,soil=custom,crop=maizeDH02,initial_water_content=initWC, irrigation_management=rainfed) # create model
model.run_model(till_termination=True) # run model till the end
outputs_water_flux.append(model._outputs.water_flux)
outputs_water_storage.append(model._outputs.water_storage)
outputs_crop_growth.append(model._outputs.crop_growth)
outputs_final_stats.append(model._outputs.final_stats) # save results


for i in range(len(outputs_final_stats)):
    outputs_final = pd.DataFrame(outputs_final_stats[i])[['Harvest Date (YYYY/MM/DD)', 'Yield (tonne/ha)']]


for i in range(len(outputs_water_flux)):
    modelled_water_flux = pd.DataFrame(outputs_water_flux[i][['dap', 'Es', 'Tr']])

modelled_water_flux['modelled_ET'] = modelled_water_flux.Es + modelled_water_flux.Tr

    
for i in range(len(outputs_water_storage)):
    modelled_water_storage = pd.DataFrame(outputs_water_storage[i][['th1', 'th2', 'th3', 'th5']])

for i in range(len(outputs_crop_growth)):
    modelled_crop_growth = pd.DataFrame(outputs_crop_growth[i][['canopy_cover', 'biomass', 'harvest_index', 'harvest_index']])

############
# Plotting #
############

out = outputs_water_storage[0]
out['ET_mod'] = modelled_water_flux.modelled_ET
out['ET_PM'] = dd['ReferenceET'].values
out['Precipitation'] = dd['Precipitation'].values

plt.show()

outputs_final['Date'] = outputs_final['Harvest Date (YYYY/MM/DD)']

outputs_final = outputs_final.set_index(pd.to_datetime(outputs_final.Date))

outputs_final['year'] = outputs_final.index.year

crop_yield = pd.DataFrame(outputs_final['year'])

crop_yield['yield'] = outputs_final['Yield (tonne/ha)']

crop_yield.to_csv('centennial_yield_1.csv', index=False)