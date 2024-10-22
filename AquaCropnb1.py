
import os

os.environ['DEVELOPMENT'] = 'DEVELOPMENT'

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# specify filepath to weather file (either locally or imported to colab)
# filepath= 'YOUR_WEATHER_FILE.TXT'

# weather_data = prepare_weather(filepath)
# weather_data

# locate built in weather file

filepath=get_filepath('tunis_climate.txt')

weather_data = prepare_weather(filepath)

#soil

sandy_loam = Soil(soil_type='SandyLoam')

#crop

wheat = Crop('Wheat', planting_date='10/01')

# Initial water content

# InitWC = InitialWaterContent(value=['FC'])

# Same default WC but displaying all parameter values:

InitWC = InitialWaterContent(wc_type = 'Prop',
                               method = 'Layer',
                               depth_layer= [1],
                               value = ['FC'])

# combine into aquacrop model and specify start and end simulation date

model = AquaCropModel(sim_start_time=f'{1979}/10/01',
                      sim_end_time=f'{1985}/05/30',
                      weather_df=weather_data,
                      soil=sandy_loam,
                      crop=wheat,
                      initial_water_content=InitWC)

model.run_model(till_termination=True)

# combine into aquacrop model and specify start and end simulation date

model_clay = AquaCropModel(sim_start_time=f'{1979}/10/01',
                      sim_end_time=f'{1985}/05/30',
                      weather_df=weather_data,
                      soil=Soil('Clay'),
                      crop=wheat,
                      initial_water_content=InitWC)

model_clay.run_model(till_termination=True)

# collate our seasonal yields so we can visualize our results.

names=['Sandy Loam','Clay']

#combine our two output files
dflist=[model._outputs.final_stats,
        model_clay._outputs.final_stats] 



outlist=[]
for i in range(len(dflist)): # go through our two output files
    temp = pd.DataFrame(dflist[i]) #['Yield (tonne/ha)'] # extract the seasonal yield data
    
    temp['label']=names[i] # add the soil type label
   
    outlist.append(temp) # save processed results


# combine results

all_outputs = pd.concat(outlist,axis=0)


# visualize and compare the yields from the two different soil types.

#create figure
fig,ax=plt.subplots(1,1,figsize=(10,7),)

# create box plot
sns.boxplot(data=all_outputs,x='label',y='Yield (tonne/ha)',ax=ax,)

# labels and font sizes
ax.tick_params(labelsize=15)
ax.set_xlabel(' ')
ax.set_ylabel('Yield (tonne/ha)',fontsize=18)

# Display plots

plt.show()