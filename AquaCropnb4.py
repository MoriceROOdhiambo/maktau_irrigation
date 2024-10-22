
import os

os.environ['DEVELOPMENT'] = 'DEVELOPMENT'

# Notebook 4: Irrigation demands under different climate change scenarios.
# In this notebook, we show how AquaCrop-OSPy can be used to simulate impacts of climate change on crop production and irrigation water demands
# Climate change is a major driver of production and water scarcity for agriculture globally, and data generated can thus play an important role in designing effective adaptation measures to support farmers and rural economies.

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent,CO2, IrrigationManagement
from aquacrop.utils import get_filepath, prepare_lars_weather, select_lars_wdf

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# imports the baseline data from Champion, Nebraska

# get baseline lars data

lars_fp=get_filepath('CP.dat')
lars_base=prepare_lars_weather(lars_fp,-1,False,order=['year', 'jday', 'maxTemp', 'minTemp', 'precip','rad',])

# For each combination of climate scenario and projection period, read in climate data and save the weather DataFrame.

rcp_list = [45,85]
yr_list = [2030,2050,2070]
yr_range_list = ['2021-2040','2041-2060','2061-2080']

label_list=[]
wdf_list = []
all_year_list=[]
for yr,yr_range in zip(yr_list,yr_range_list):
    for rcp in rcp_list:
        wdf =prepare_lars_weather(get_filepath(f'CP_EC-EARTH[CP,RCP{rcp},{yr_range}]WG.dat'), yr,True,["simyear","jday","minTemp","maxTemp","precip","rad"])
        wdf_list.append(wdf)
        label_list.append(f'RCP{rcp/10},\n{yr_range}')
        all_year_list.append(yr)

# co2 concentrations for the scenarios listed in order

co2_list = [435,449,487,541,524,677]

# Now that all the climate data is ready, we can define our crop, soil, initial water content and irrigation management.
# In this example we will use the AquaCrop default Maize crop in calendar day mode. The reason for using calendar day mode is so that the growing season will be the same length in each scenario.
# We choose a Clay loam soil starting at Field Capacity, and an irrigation policy that irrigates if the soil drops below 70% total available water (essentially meeting full crop-water requirements).

crop=Crop('Maize',planting_date='05/01', CalendarType=1,Emergence = 6,Senescence=107, MaxRooting=108,Maturity=132,HIstart=66, Flowering=13,YldForm=61,CDC=0.117,CGC=0.163)
          
soil=Soil('ClayLoam')

init_wc = InitialWaterContent() # default is field capacity

irrmngt=IrrigationManagement(1,SMT=[70]*4)

# Run the simulation for the baseline period and save the yields and total irrigation

#run for baseline scenario
model=AquaCropModel('1982/05/01','2018/10/30',lars_base,soil,crop, init_wc,irrigation_management=irrmngt)

model.run_model(till_termination=True)

baseline_yields = list(model.get_simulation_results()['Yield (tonne/ha)'].values)
baseline_tirrs = list(model.get_simulation_results()['Seasonal irrigation (mm)'].values)
baseline_labels = ['Baseline']*len(baseline_tirrs)

# Define run_gen_model function that runs AquaCrop-OSPy for all 100 generated years of a climate scenario and future period (e.g. 2050 at RCP4.5), saving the yields and total irrigation.

def run_gen_model(all_wdf,co2conc,year):
    gen_yields=[]
    gen_tirrs=[]
    for i in range(100):
        wdf = select_lars_wdf(all_wdf,i+1)

        co2 = CO2(constant_conc=True)
        co2.current_concentration = co2conc
        
        model=AquaCropModel(f'{year}/05/01',f'{year}/10/30',wdf,soil,crop, InitialWaterContent(),irrigation_management=irrmngt,co2_concentration=co2)

        model.run_model(till_termination=True)

        gen_yields.append(model.get_simulation_results()['Yield (tonne/ha)'].mean())
        gen_tirrs.append(model.get_simulation_results()['Seasonal irrigation (mm)'].mean())

    return gen_yields,gen_tirrs

# For each combination of climate scenario and projection period, run AquaCrop-OSPy and save results.

all_ylds = []
all_tirrs = []
all_labels = []
for i in (range(6)):
    year = all_year_list[i]
    wdf = wdf_list[i]
    co2 = co2_list[i]
    label=label_list[i]

    yld_list,tirr_list = run_gen_model(wdf,co2,year)

    all_ylds.extend(yld_list)
    all_tirrs.extend(tirr_list)
    all_labels.extend([label]*len(yld_list))

# Combine projection results with baseline.

all_ylds = baseline_yields+all_ylds
all_tirrs = baseline_tirrs+all_tirrs
all_labels = baseline_labels+all_labels

df = pd.DataFrame([all_ylds,all_tirrs,all_labels]).T

df.columns = ['Yield','Tirr','Label']

# create figure 
fig,ax=plt.subplots(2,1,figsize=(12,14))

# create box plots
sns.boxplot(data=df,x='Label',y='Yield',ax=ax[0])
sns.boxplot(data=df,x='Label',y='Tirr',ax=ax[1])

# labels and fontsize

ax[0].tick_params(labelsize=15)
ax[0].set_xlabel(' ')
ax[0].set_ylabel('Yield (t/ha)',fontsize=18)

ax[1].tick_params(labelsize=15)
ax[1].set_xlabel(' ')
ax[1].set_ylabel('Total Irrigation (ha-mm)',fontsize=18)

# Appendix: Precipitation and ET changes

all_precip = []
all_et = []
all_new_labels = []

for year in range(1982,2019):

    #run for baseline scenario
    wdf = lars_base[lars_base.Date>f'{year}-05-01']

    wdf = wdf[wdf.Date<f'{year}-10-31']

    all_precip.extend([wdf.Precipitation.mean()])

    all_et.extend([wdf.ReferenceET.mean()])

    all_new_labels.extend(['baseline'])

for i in range(6):

    year = all_year_list[i]
    wdf = wdf_list[i]
    co2 = co2_list[i]
    label=label_list[i]

    wdf = wdf[wdf.Date>f'{year}-05-01']
    wdf = wdf[wdf.Date<f'{year}-10-31']

    #print(wdf.ReferenceET.mean())

    precip_list = list(wdf.groupby('simyear').mean().Precipitation.values)
    et_list = list(wdf.groupby('simyear').mean().ReferenceET.values)


    all_precip.extend(precip_list)
    all_et.extend(et_list)
    all_new_labels.extend([label]*len(et_list))

df = pd.DataFrame([all_precip,all_et,all_new_labels]).T
df.columns = ['precip','et','Label']

# create figure 
fig,ax=plt.subplots(2,1,figsize=(12,14))

# create box plots
sns.boxplot(data=df,x='Label',y='precip',ax=ax[0])
sns.boxplot(data=df,x='Label',y='et',ax=ax[1])

# labels and fontsize

ax[0].tick_params(labelsize=15)
ax[0].set_xlabel(' ')
ax[0].set_ylabel('Precipitation (mm/ha)',fontsize=18)

ax[1].tick_params(labelsize=15)
ax[1].set_xlabel(' ')
ax[1].set_ylabel('ETo (mm/ha)',fontsize=18)

plt.show()