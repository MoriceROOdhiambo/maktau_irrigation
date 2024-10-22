
# In this notebook, we show how AquaCrop-OSPy can be used to explore impacts of different irrigation management strategies on water use and crop yields.
# The example workflow below shows how different irrigation management practices can be defined in the model, and resulting impacts on water use productivity explored to support efficient irrigation scheduling and planning decisions.

import os

os.environ['DEVELOPMENT'] = 'DEVELOPMENT'

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, CO2, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
supress_warnings = True
warnings.filterwarnings('ignore') 

path = get_filepath('champion_climate.txt')
wdf = prepare_weather(path)

sim_start = '1982/05/01'
sim_end = '2018/10/30'

soil= Soil('SandyLoam')

crop = Crop('Maize',planting_date='05/01')

initWC = InitialWaterContent(value=['FC'])

"""
Irrigation management parameters are selected by creating an IrrigationManagement object. With this class we can specify a range of different irrigation management strategies.

The 6 different strategies can be selected using the IrrMethod argument when creating the class. These strategies are as follows:

IrrMethod=0: Rainfed (no irrigation)
IrrMethod=1: Irrigation is triggered if soil water content drops below a specified threshold (or four thresholds representing four major crop growth stages (emergence, canopy growth, max canopy, senescence).
IrrMethod=2: Irrigation is triggered every N days
IrrMethod=3: Predefined irrigation schedule
IrrMethod=4: Net irrigation (maintain a soil-water level by topping up all compartments daily)
IrrMethod=5: Constant depth applied each day



"""

# define labels to help after

labels=[]

outputs=[]

for smt in range(0,110,20):
    #crop.Name = str(smt) # add helpfull label
    labels.append(str(smt))
    irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[smt]*4) # specify irrigation management
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=irr_mngt) # create model
    model.run_model(till_termination=True) # run model till the end
    outputs.append(model._outputs.final_stats) # save results



dflist=outputs

labels[0]='Rainfed'

print(outputs)
labels[0]='Rainfed'
print(labels)
outlist=[]

for i in range(len(dflist)):
    temp = pd.DataFrame(dflist[i][['Yield (tonne/ha)',
                                   'Seasonal irrigation (mm)']])
    temp['label']=labels[i]
    outlist.append(temp)


all_outputs = pd.concat(outlist,axis=0)

# combine all results
results=pd.concat(outlist)

# create figure consisting of 2 plots
fig,ax=plt.subplots(2,1,figsize=(10,14))

# create two box plots
sns.boxplot(data=results,x='label',y='Yield (tonne/ha)',ax=ax[0])
sns.boxplot(data=results,x='label',y='Seasonal irrigation (mm)',ax=ax[1])

# labels and font sizes
ax[0].tick_params(labelsize=15)
ax[0].set_xlabel('Soil-moisture threshold (%TAW)',fontsize=18)
ax[0].set_ylabel('Yield (t/ha)',fontsize=18)

ax[1].tick_params(labelsize=15)
ax[1].set_xlabel('Soil-moisture threshold (%TAW)',fontsize=18)
ax[1].set_ylabel('Total Irrigation (ha-mm)',fontsize=18)

plt.legend(fontsize=18)

plt.show()
