
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

# define irrigation management

#The first strategy we will test is rainfed growth (no irrigation).

rainfed = IrrigationManagement(irrigation_method=0)

# The second strategy triggers irrigation if the root-zone water content drops below an irrigation threshold.
# There are 4 thresholds corresponding to four main crop growth stages (emergence, canopy growth, max canopy, canopy senescence).
# The quantity of water applied is given by min(depletion,MaxIrr) where MaxIrr can be specified when creating an IrrMngtClass.

threshold4_irrigate = IrrigationManagement(irrigation_method=1,SMT=[40,60,70,30]*4)

# The third strategy irrigates every IrrInterval days where the quantity of water applied is given by min(depletion,MaxIrr)
# where MaxIrr can be specified when creating an IrrMngtClass.
# irrigate every 7 days

interval_7 = IrrigationManagement(irrigation_method=2,IrrInterval=7)

# The fourth strategy irrigates according to a predefined calendar.
# This calendar is defined as a pandas DataFrame and this example, we will create a calendar that irrigates on the first Tuesday of each month.

all_days = pd.date_range(sim_start,sim_end) # list of all dates in simulation period

new_month=True
dates=[]
# iterate through all simulation days
for date in all_days:
    #check if new month
    if date.is_month_start:
        new_month=True

    if new_month:
        # check if tuesday (dayofweek=1)
        if date.dayofweek==1:
            #save date
            dates.append(date)
            new_month=False

depths = [25]*len(dates) # depth of irrigation applied
schedule=pd.DataFrame([dates,depths]).T # create pandas DataFrame
schedule.columns=['Date','Depth'] # name columns

irrigate_schedule = IrrigationManagement(irrigation_method=3,schedule=schedule)

#The fifth strategy is net irrigation. This keeps the soil-moisture content above a specified level.
# This method differs from the soil moisture thresholds (second strategy) as each compartment is filled to field capacity,
# instead of water starting above the first compartment and filtering down.
# In this example the net irrigation mode will maintain a water content of 70% total available water.

net_irrigation = IrrigationManagement(irrigation_method=4,NetIrrSMT=70)

# define labels to help after
labels=['rainfed','four thresholds','interval','schedule','net']
strategies = [rainfed,threshold4_irrigate,interval_7,irrigate_schedule,net_irrigation]

outputs=[]
for i,irr_mngt in enumerate(strategies): # for both irrigation strategies...
    crop.Name = labels[i] # add helpfull label
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=irr_mngt) # create model
    model.run_model(till_termination=True) # run model till the end
    outputs.append(model._outputs.final_stats) # save results

#The final strategy to show is for a custom irrigation strategy.
# This is one of the key features of AquaCrop-OSPy as users can define an a complex irrigation strategy that incorperates any external data,
# code bases or machine learning models.
# To showcase this feature, we will define a function that will irrigate according to the follwong logic:
# 1) There will be no rain over the next 10 days -> Irrigate 10mm
# 2) There will be rain in the next 10 days but the soil is over 70% depleted -> Irrigate 10mm
# 3) Otherwise -> No irrigation

# function to return the irrigation depth to apply on next day
def get_depth(model):    
    t = model._clock_struct.time_step_counter # current timestep
    # get weather data for next 7 days
    weather10 = model._weather[t+1:min(t+10+1,len(model._weather))]
    # if it will rain in next 7 days
    if sum(weather10[:,2])>0:
        # check if soil is over 70% depleted
        if t>0 and model._init_cond.depletion/model._init_cond.taw > 0.7:
            depth=10
        else:
            depth=0
    else:
        # no rain for next 10 days
        depth=10

    return depth

# create model with IrrMethod= Constant depth
crop.Name = 'weather' # add helpfull label

model = AquaCropModel(sim_start,sim_end,wdf,soil,crop,initial_water_content=initWC,
                      irrigation_management=IrrigationManagement(irrigation_method=5,)) 

model._initialize()

while model._clock_struct.model_is_finished is False:    
    # get depth to apply
    depth=get_depth(model)
    
    model._param_struct.IrrMngt.depth=depth

    model.run_model(initialize_model=False)

outputs.append(model._outputs.final_stats) # save results
labels.append('weather')

dflist=outputs
outlist=[]
for i in range(len(dflist)):
    temp = pd.DataFrame(dflist[i][['Yield (tonne/ha)','Seasonal irrigation (mm)']])
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
ax[0].set_xlabel(' ')
ax[0].set_ylabel('Yield (t/ha)',fontsize=18)

ax[1].tick_params(labelsize=15)
ax[1].set_xlabel(' ')
ax[1].set_ylabel('Total Irrigation (ha-mm)',fontsize=18)

plt.legend(fontsize=18)


plt.show()
