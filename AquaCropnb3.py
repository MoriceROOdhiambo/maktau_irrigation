
import os

os.environ['DEVELOPMENT'] = 'DEVELOPMENT'

# Notebook 3: Developing and optimizing irrigation stratgeies. Optimizing soil-moisture thresholds. 
# During the growing season, if the soil-moisture content drops below the threshold, irrigation is applied to refill the soil profile back to field capacity subject to a maximum irrigation depth

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, CO2, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
 


path = get_filepath('champion_climate.txt')
wdf = prepare_weather(path)

# Define a function called run_model that creates and runs an AquaCrop model (just like in the previous notebooks), and returns the final output.


def run_model(smts,max_irr_season,year1,year2):
    """
    funciton to run model and return results for given set of soil moisture targets
    """

    maize = Crop('Maize',planting_date='05/01') # define crop
    loam = Soil('ClayLoam') # define soil
    init_wc = InitialWaterContent(wc_type='Pct',value=[70]) # define initial soil water conditions

    irrmngt = IrrigationManagement(irrigation_method=1,SMT=smts,MaxIrrSeason=max_irr_season) # define irrigation management

    # create and run model
    model = AquaCropModel(f'{year1}/05/01',f'{year2}/10/31',wdf,loam,maize, irrigation_management=irrmngt,initial_water_content=init_wc)

    model.run_model(till_termination=True)
    return model.get_simulation_results()

# Define evaluate will act as a reward function for the optimization library to optimize. Inside this function we run the model and return the reward (in this case the average yield).

def evaluate(smts,max_irr_season,test=False):
    """
    funciton to run model and calculate reward (yield) for given set of soil moisture targets
    """
    # run model
    out = run_model(smts,max_irr_season,year1=2016,year2=2018)
    # get yields and total irrigation
    yld = out['Yield (tonne/ha)'].mean()
    tirr = out['Seasonal irrigation (mm)'].mean()

    reward=yld

    # return either the negative reward (for the optimization)
    # or the yield and total irrigation (for analysis)
    if test:
        return yld,tirr,reward
    else:
        return -reward
    
# Define get_starting_point that chooses a set of random irrigation strategies and evaluates them to give us a good starting point for our optimization. (Since we are only using a local minimization function this will help get a good result)

def get_starting_point(num_smts,max_irr_season,num_searches):
    """
    find good starting threshold(s) for optimization
    """

    # get random SMT's
    x0list = np.random.rand(num_searches,num_smts)*100
    rlist=[]
    # evaluate random SMT's
    for xtest in x0list:
        r = evaluate(xtest,max_irr_season,)
        rlist.append(r)

    # save best SMT
    x0=x0list[np.argmin(rlist)]
    
    return x0

# Define optimize that uses the scipy.optimize.fmin optimization package to find yield maximizing irrigation strategies for a maximum seasonal irrigation limit.

def optimize(num_smts,max_irr_season,num_searches=100):
    """ 
    optimize thresholds to be profit maximising
    """
    # get starting optimization strategy
    x0=get_starting_point(num_smts,max_irr_season,num_searches)
    # run optimization
    res = fmin(evaluate, x0,disp=0,args=(max_irr_season,))
    # reshape array
    smts= res.squeeze()
    # evaluate optimal strategy
    return smts

# For a range of maximum seasonal irrigation limits (0-450mm), find the yield maximizing irrigation schedule.

opt_smts=[]
yld_list=[]
tirr_list=[]

for max_irr in range(0,500,50):    

    # find optimal thresholds and save to list
    smts=optimize(4,max_irr)
    opt_smts.append(smts)

    # save the optimal yield and total irrigation
    yld,tirr,_=evaluate(smts,max_irr,True)
    yld_list.append(yld)
    tirr_list.append(tirr)

# Visualize the optimal yield and total irrigation, creating a crop-water production function.

# create plot
fig,ax=plt.subplots(1,1,figsize=(13,8))

# plot results
ax.scatter(tirr_list,yld_list)
ax.plot(tirr_list,yld_list)

# labels
ax.set_xlabel('Total Irrigation (ha-mm)',fontsize=18)
ax.set_ylabel('Yield (tonne/ha)',fontsize=18)
ax.set_xlim([-20,600])
ax.set_ylim([2,15.5])

# annotate with optimal thresholds
bbox = dict(boxstyle="round",fc="1")
offset = [15,15,15, 15,15,-125,-100,  -5, 10,10]
yoffset= [0,-5,-10,-15, -15,  0,  10,15, -20,10]
for i,smt in enumerate(opt_smts):
    smt=smt.clip(0,100)
    ax.annotate('(%.0f, %.0f, %.0f, %.0f)'%(smt[0],smt[1],smt[2],smt[3]),
                (tirr_list[i], yld_list[i]), xytext=(offset[i], yoffset[i]), textcoords='offset points',
                bbox=bbox,fontsize=12)
    
plt.show()