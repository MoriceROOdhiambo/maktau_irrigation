"""Run aquacrop in Maktau


"""

import os
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'

import numpy as np
import matplotlib.pylab as plt
import pandas as pd

#import seaborn as sns

import atm

import warnings

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

supress_warnings = True

warnings.filterwarnings('ignore') 

#from SeqMetrics import nse, rmse, RegressionMetrics, plot_metrics, agreement_index, r2, mae, nrmse, pbias, mean_bias_error

#from scipy import stats

#################################################
# Read Maktau AWS data and prepare model inputs #
#################################################

met = pd.read_csv('Maktau_AWS_1h.dat')

met = met.set_index(pd.to_datetime(met['Unnamed: 0']))
met.index.name = ''

smplot = pd.read_csv("maktau_irrigation_vwc.csv")

smplot = smplot.set_index(pd.to_datetime(smplot.Date))

# Incoming global radiation scaled to attain Net radiation

met['net_mod'] = 0.78*met.SlrW_Avg
met['vpd'] = atm.calc_vpd(met.AirTC_Avg, met.RH)
met['soilf'] = met.filter(regex='Flux_corr').mean(axis=1)
met['daytime'] = met.SlrW_Avg > 10
met['Rain_mm_Tot_corr'] = 1.07*met.Rain_mm_Tot

met['et_pm'] = atm.et_penman(met.WS_ms_S_WVT, 90, met.vpd, met.AirTC_Avg, met.net_mod, met['soilf'], met.daytime, timestep=3600)

per = slice('20240403','20240815')
dat = met[per]
dd = dat.resample("D").mean()

dd['Precipitation'] = dat.resample("D").Rain_mm_Tot_corr.sum()
dd['ReferenceET'] = dat.resample("D").et_pm.sum()
dd['MinTemp'] = dat.resample("D").AirTC_Avg.min()
dd['MaxTemp'] = dat.resample("D").AirTC_Avg.max()
dd['Date'] = dd.index.values

# Model input

wdf = dd[['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET' , 'Date']]

wdf = wdf.reset_index()

wdf = wdf.drop([''], axis=1)

print(wdf)

#######################
# Aquacrop simulation #
#######################

# Soil

custom = Soil('custom', dz=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

custom.add_layer(thickness=0.1, thWP=0.06, thFC=0.17, thS=0.45, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.1, thWP=0.08, thFC=0.20, thS=0.42, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.1, thWP=0.08, thFC=0.22, thS=0.42, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.2, thWP=0.10, thFC=0.23, thS=0.42, Ksat=1000, penetrability=100)

custom.add_layer(thickness=0.5, thWP=0.10, thFC=0.23, thS=0.42, Ksat=1000, penetrability=100)

# Initial water content

initWC = InitialWaterContent(wc_type='Prop', method='Layer', depth_layer=[1,2,3,4,5], value=['FC','FC','FC','FC','FC'])


# Soil moisture in the plot

sm_dd = smplot.resample("H").mean()

sm_dd.to_csv('sm_dd.csv')

# AWS soil moisture

period = slice('20240403','20240815')

dd_measured = dd[period]

measured_met = dd_measured[['VWC_10cm_corr', 'VWC_30cm_corr', 'VWC_50cm_corr']]

# Crop

plant_row=11
plants_per_row=15
plant_population=(plant_row*plants_per_row)*100 # 100*100=10,000m**2=1ha

               
maizePH1=Crop('Maize', planting_date='03/17', PlantMethod=1, CalenderType=1, CGC=0.163, CDC=0.117,
               Emergence=6, Flowering=13, HIstart=66, YldForm=61, MaxRooting=108, Senescence=107, Maturity=132, HI0=0.48,
               Zmin=0.3, Zmax=1.0, PlantPop=39200, WP=33.7, CCx=0.88,                
               fshape_w1=2.9,fshape_w2=6.0, fshape_w3=2.7)


# Irrigation

plot_area=100 #m**2
emmiters=156
emmiter_discharge=0.6 #L/hr
irr_event_time=2 #hr
wetted_area_radius=0.15
wetted_area=np.pi*wetted_area_radius**2
total_wetted_area=wetted_area*emmiters
wetted_surface=(total_wetted_area/plot_area)*100
irr_event_vol_emitter=emmiter_discharge*irr_event_time
irr_depth = (((irr_event_vol_emitter*emmiters)/1000)/total_wetted_area)*1000

print(irr_depth)

irr_mngt = IrrigationManagement(irrigation_method=2,IrrInterval=3,WetSurf=wetted_surface,AppEff=90,MaxIrr=7)

'''
maize water stress level soil moisture content < 0.125 m3/m3
Depletion = 68% at 0.3m
Depletion = 80% at 0.5m

#custom irrigation strategy.
# To showcase this feature, we will define a function that will irrigate according to the follwing logic:
# 1) There will be no rain over the next 10 days -> Irrigate 10mm
# 2) There will be rain in the next 10 days but the soil is over 70% depleted -> Irrigate 10mm
# 3) Otherwise -> No irrigation


# function to return the irrigation depth to apply on next day
def get_depth(model):    
    t = model._clock_struct.time_step_counter # current timestep
    # get weather data for next n days
    weather10 = model._weather[t+1:min(t+3+1,len(model._weather))]
    # if it will rain in next n days
    if sum(weather10[:,2])>0:
        depth=0
    else:
        # no rain for next 10 days
        depth=irr_depth

    return depth
'''

# Output labels

outputs_water_flux=[]
outputs_water_storage=[]
outputs_crop_growth=[]
outputs_final_stats=[]

#model = AquaCropModel(sim_start,sim_end,wdf,soil,crop,initial_water_content=initWC,
#                      irrigation_management=IrrigationManagement(irrigation_method=5,))

sim_start = '2024/04/03'

sim_end = '2024/08/15'

model = AquaCropModel(sim_start,sim_end,wdf,soil=custom,crop=maizePH1,initial_water_content=initWC,
                      irrigation_management=irr_mngt) # create model 

'''
model._initialize()

while model._clock_struct.model_is_finished is False:    
    # get depth to apply
    depth=get_depth(model)
    
    model._param_struct.IrrMngt.depth=depth

    model.run_model(initialize_model=False)

'''
model.run_model(till_termination=True) # run model till the end
outputs_water_flux.append(model._outputs.water_flux)
outputs_water_storage.append(model._outputs.water_storage)
outputs_crop_growth.append(model._outputs.crop_growth)
outputs_final_stats.append(model._outputs.final_stats) # save results


for i in range(len(outputs_final_stats)):
    outputs_final = pd.DataFrame(outputs_final_stats[i])

print(outputs_final)


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

'''

out = outputs_water_storage[0].iloc[:-75,:]
out['swc10'] = measured_met.VWC_30cm_corr.iloc[:-75].values
out['swc30'] = measured_met.VWC_30cm_corr.iloc[:-75].values
out['swc50'] = measured_met.VWC_50cm_corr.iloc[:-75].values

out['ET_mod'] = modelled_water_flux.modelled_ET.iloc[:-75]
out['ET_measured'] = modelled_water_flux.modelled_ET.iloc[:-75]
#out['ET_measured'] = measured_met.evap_corr_mm.iloc[:-2].values
out['ET_PM'] = dd['ReferenceET'][period].iloc[:-75].values
out['Precipitation'] = dd['Precipitation'][period].iloc[:-75].values
out['model_canopy_cover'] = modelled_crop_growth.canopy_cover.iloc[:-75].values
out['ndvi_maize'] = dd['ndvi_maize'][period].iloc[:-75].values
out['measured_canopy_cover'] = ((105.427*out.ndvi_maize) + (-6.501))/100

print(outputs_final['Yield (tonne/ha)'])

#########################################################################

# Evapotranspiration

fig, ax = plt.subplots(figsize=(8,8))
out.ET_mod.plot(ax=ax, color='C0', linewidth='3')
#out.ET_measured.plot(ax=ax, color='C3', linewidth='3')
out.ET_PM.plot(ax=ax, color='C1', linewidth='3')

# Add labels and title (optional)
ax.tick_params(labelsize=18, width='3')
ax.set_xlabel('DAS', fontsize=20, weight='bold')
ax.set_ylabel('ET (mm)', fontsize=20, weight='bold')
ax.set_title('Rainfed 2014', fontsize=20, weight='bold')
ax.legend(loc='best', fontsize=16)
plt.tight_layout()
#print(plt.style.available)
#plt.style.use('style')

# Regression
fig, ax = plt.subplots(figsize=(8,8)) 
sns.scatterplot(data=out, x='ET_measured', y='ET_mod', s=50, color=".1")
ax.set_title('Rainfed 2014', fontsize=20, weight='bold')
ax.set_xlabel('Measured ET (mm)', fontsize=20, weight='bold')
ax.set_ylabel('Modelled ET (mm)', fontsize=20, weight='bold')
ax.tick_params(labelsize=18, width='3')
res = stats.linregress(out.ET_measured, out.ET_mod)
line = res.slope * out.ET_measured + res.intercept
ax.plot(out.ET_measured, line, color='C3', linestyle='solid', linewidth='3')

# Draw the 1:1 line through the origin
#min_val = (50, 50) 
#max_val = (250, 250)
#ax.plot([min_val, max_val], [min_val, max_val], color='0.0', linestyle='dashed')
ax.axline((0,0), slope = 1, color='Firebrick', linestyle='solid')

# Set both x- and y-axis limits
#ax.axis([50, 250, 50, 250])
ax.text(6, 1, f' y = {res.intercept:.2f} + {res.slope:.2f}x', fontsize=20, weight='bold')
plt.tight_layout()

#Student's t-test

"Student's t test for Ho: slope = 1 and offset = 0"

ET_t_statistic_slope = (res.slope - 1)/res.stderr

ET_t_statistic_intercept = res.intercept/res.intercept_stderr

n = len(out.ET_measured)

ET_p_value_slope = 2 * (1 - stats.t.cdf(np.abs(ET_t_statistic_slope), n - 2))

ET_p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(ET_t_statistic_intercept), n - 2))

print(f'ET_p_value_slope = {ET_p_value_slope}, ET_p_value_intercept = {ET_p_value_intercept}')

# Statistical indicators

R2_ET = r2(out.ET_measured, out.ET_mod)
RMSE_ET = rmse(out.ET_measured, out.ET_mod)
NRMSE_ET = nrmse(out.ET_measured, out.ET_mod)
EF_ET = nse(out.ET_measured, out.ET_mod)
d_index_ET=agreement_index(out.ET_measured, out.ET_mod)
PBIAS_ET = pbias(out.ET_measured, out.ET_mod)
CRM_ET = (np.sum(out.ET_measured) - np.sum(out.ET_mod))/(np.sum(out.ET_measured))
MAE_ET = mae(out.ET_measured, out.ET_mod)
MBE_ET = mean_bias_error(out.ET_measured, out.ET_mod)

print(f"R2_ET = {R2_ET},RMSE_ET = {RMSE_ET}, NRMSE_ET = {NRMSE_ET}, EF_ET = {EF_ET},
       d-index_ET = {d_index_ET}, PBIAS_ET = {PBIAS_ET}, CRM_ET = {CRM_ET}, MBE_ET = {MBE_ET}")

###################################################################################

# Canopy

fig, ax = plt.subplots(figsize=(8,8))
out.model_canopy_cover.plot(ax=ax, color='C0', linewidth='3')
ax.legend(loc='best', fontsize=16)
plt.tight_layout()

# Regression

fig, ax = plt.subplots(figsize=(8,8)) 
sns.scatterplot(data=out, x='measured_canopy_cover', y='model_canopy_cover', s=50, color=".1")
ax.set_title('Rainfed 2014', fontsize=20, weight='bold')
ax.set_xlabel('Measured CC (%)', fontsize=20, weight='bold')
ax.set_ylabel('Modelled CC (%)', fontsize=20, weight='bold')
ax.tick_params(labelsize=18, width='3')
res = stats.linregress(out.measured_canopy_cover, out.model_canopy_cover)
line = res.slope * out.measured_canopy_cover + res.intercept
ax.plot(out.measured_canopy_cover, line, color='C3', linestyle='solid', linewidth='3')

# Draw the 1:1 line through the origin
#min_val = (50, 50) 
#max_val = (250, 250)
#ax.plot([min_val, max_val], [min_val, max_val], color='0.0', linestyle='dashed')
ax.axline((0,0), slope = 1, color='Firebrick', linestyle='solid')

# Set both x- and y-axis limits
#ax.axis([50, 250, 50, 250])
ax.text(6, 1, f' y = {res.intercept:.2f} + {res.slope:.2f}x', fontsize=20, weight='bold')
plt.tight_layout()

#Student's t-test

"Student's t test for Ho: slope = 1 and offset = 0"

CC_t_statistic_slope = (res.slope - 1)/res.stderr

CC_t_statistic_intercept = res.intercept/res.intercept_stderr

n = len(out.measured_canopy_cover)

CC_p_value_slope = 2 * (1 - stats.t.cdf(np.abs(CC_t_statistic_slope), n - 2))

CC_p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(CC_t_statistic_intercept), n - 2))

print(f'CC_p_value_slope = {CC_p_value_slope}, CC_p_value_intercept = {CC_p_value_intercept}')

# Statistical indicators

R2_CC = r2(out.measured_canopy_cover, out.model_canopy_cover)
RMSE_CC = rmse(out.measured_canopy_cover, out.model_canopy_cover)
NRMSE_CC = nrmse(out.measured_canopy_cover, out.model_canopy_cover)
EF_CC = nse(out.measured_canopy_cover, out.model_canopy_cover)
d_index_CC=agreement_index(out.measured_canopy_cover, out.model_canopy_cover)
PBIAS_CC = pbias(out.measured_canopy_cover, out.model_canopy_cover)
CRM_CC = (np.sum(out.measured_canopy_cover) - np.sum(out.model_canopy_cover))/(np.sum(out.measured_canopy_cover))
MAE_CC = mae(out.measured_canopy_cover, out.model_canopy_cover)
MBE_CC = mean_bias_error(out.measured_canopy_cover, out.model_canopy_cover)

print(f"R2_CC = {R2_CC},RMSE_CC = {RMSE_CC}, NRMSE_CC = {NRMSE_CC}, EF_CC = {EF_CC},
       d-index_CC = {d_index_CC}, PBIAS_CC = {PBIAS_CC}, CRM_CC = {CRM_CC}, MBE_CC = {MBE_CC}")

####################################################################################

# Soil Water Content

fig, ax = plt.subplots(3, figsize=(8,8))
out.th1.plot(ax=ax[0], color='C0', linewidth='3')
out.swc10.plot(ax=ax[0], color='C3', linewidth='3')
out.th3.plot(ax=ax[1], color='C0', linewidth='3')
out.swc30.plot(ax=ax[1], color='C3', linewidth='3')
out.th5.plot(ax=ax[2], color='C0', linewidth='3')
out.swc50.plot(ax=ax[2], color='C3', linewidth='3')
fig.suptitle('Rainfed 2014', fontsize=20, weight='bold')
plt.tight_layout()


for i in ax:
    i.legend(loc='best', fontsize=16)

for ax in ax:
    ax.set_xlabel('DAS', fontsize=20, weight='bold')
    ax.set_ylabel('SWC (mm)', fontsize=20, weight='bold')
    ax.tick_params(labelsize=18, width='3')

#

def calculate_sws(swc, z):
    """
    Calculate the total Soil Water Storage (SWS).
    
    :param moisture_contents: List of moisture contents (θi) for each soil layer (m³/m³)
    :param layer_depths: List of soil layer depths (zi) in mm
    :return: Total Soil Water Storage (SWS) in mm
    
    """
    if len(swc) != len(z):
        raise ValueError("The number of moisture contents must match the number of layer depths.")
    
    return np.sum(swc * z for swc, z in zip(swc, z))

swc = [out.th1, out.th3, out.th5]

z = [100, 300, 500]

model_tsw = calculate_sws(swc, z)

swc = [out.swc10, out.swc30, out.swc50]

z = [100, 300, 500]

measured_tsw = calculate_sws(swc, z)

R2 = r2(measured_tsw, model_tsw)
RMSE = rmse(measured_tsw, model_tsw)
NRMSE = nrmse(measured_tsw, model_tsw)
EF = nse(measured_tsw, model_tsw)
d_index=agreement_index(measured_tsw, model_tsw)
PBIAS = pbias(measured_tsw, model_tsw)
CRM = (np.sum(measured_tsw) - np.sum(model_tsw))/(np.sum(measured_tsw))
MAE = mae(measured_tsw, model_tsw)
MBE = mean_bias_error(measured_tsw, model_tsw)



R2_30cm = r2(out.swc30, out.th3)
RMSE_30cm = rmse(out.swc30, out.th3)
NRMSE_30cm = nrmse(out.swc30, out.th3)
result_30cm=agreement_index(out.swc30, out.th3)
EF_30cm = nse(out.swc30, out.th3)
PBIAS_30cm = pbias(out.swc30, out.th3)

R2_50cm = r2(out.swc50, out.th5)
RMSE_50cm = rmse(out.swc50, out.th5)
NRMSE_50cm = nrmse(out.swc50, out.th5)
result_50cm=agreement_index(out.swc50, out.th5)
EF_50cm = nse(out.swc50, out.th5)
PBIAS_50cm = pbias(out.swc50, out.th5)




print(f"R2 = {R2},RMSE = {RMSE}, NRMSE = {NRMSE}, EF = {EF}, d-index = {d_index}, PBIAS = {PBIAS}, CRM = {CRM}, MBE = {MBE}")


print(f"R2_30cm = {R2_30cm},RMSE_30cm = {RMSE_30cm}, NRMSE_30cm = {NRMSE_30cm}, EF_30cm = {EF_30cm}, d-index_30cm = {result_30cm}, PBIAS_30cm = {PBIAS_30cm}")

print(f"R2_50cm = {R2_50cm},RMSE_50cm = {RMSE_50cm}, NRMSE_50cm = {NRMSE_50cm}, EF_50cm = {EF_50cm}, d-index_50cm = {result_50cm}, PBIAS_50cm = {PBIAS_50cm}")

##############################################################################

# Regression

all_output = pd.DataFrame()
all_output['measured'] = measured_tsw
all_output['model'] = model_tsw

fig, ax = plt.subplots(figsize=(8,8)) 
sns.scatterplot(data=all_output, x='measured', y='model', s=50, color=".1")
ax.set_title('Rainfed 2014', fontsize=20, weight='bold')
ax.set_xlabel('Measured SWC (mm)', fontsize=20, weight='bold')
ax.set_ylabel('Modelled SWC (mm)', fontsize=20, weight='bold')
ax.tick_params(labelsize=18, width='3')
res = stats.linregress(all_output.measured, all_output.model)
line = res.slope * all_output.measured + res.intercept
ax.plot(all_output.measured, line, color='C3', linestyle='solid', linewidth='3')
# Draw the 1:1 line through the origin
min_val = (50, 50) 
max_val = (250, 250)
ax.plot([min_val, max_val], [min_val, max_val], color='0.0', linestyle='dashed')
#ax.axline((0,0), slope = 1, color='green', linestyle='solid')
# Set both x- and y-axis limits
ax.axis([50, 250, 50, 250])
ax.text(150, 75, f' y = {res.intercept:.2f} + {res.slope:.2f}x', fontsize=20, weight='bold')
plt.tight_layout()

#Student's t-test

"Student's t test for Ho: slope = 1 and offset = 0"

SWC_t_statistic_slope = (res.slope - 1)/res.stderr

SWC_t_statistic_intercept = res.intercept/res.intercept_stderr

n = len(all_output.measured)

SWC_p_value_slope = 2 * (1 - stats.t.cdf(np.abs(SWC_t_statistic_slope), n - 2))

SWC_p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(SWC_t_statistic_intercept), n - 2))

print(f'SWC_p_value_slope = {SWC_p_value_slope}, SWC_p_value_intercept = {SWC_p_value_intercept}')

####################################################################################

# Temporal evolution of SWC

fig, ax = plt.subplots(figsize=(10,7))

measured_tsw.plot(ax=ax, label = 'Measured SWC', color='C3', linewidth='3')
model_tsw.plot(ax=ax, label = 'Modelled SWC', color='C0', linewidth='3')
#ax.axhline(y=0, color='r', linestyle='-')
#ax.axhline(y=5, color='r', linestyle='-')

# Set the y-axis limits (optional)
#ax.set_xlim(-7, 124)
#ax.set_ylim(60, 250)

# Add labels and title (optional)
ax.tick_params(labelsize=18, width='3')
ax.set_xlabel('DAS', fontsize=20, weight='bold')
ax.set_ylabel('SWC (mm)', fontsize=20, weight='bold')
ax.set_title('Rainfed 2014', fontsize=20, weight='bold')
ax.legend(loc='best', fontsize=16)

# Display the plot
plt.grid(True)
plt.tight_layout()
#print(plt.style.available)
#plt.style.use('style')

############################################################################

# Water productivity

# Water productivity was determined from the ratio of the marketable crop yield [ton] to actual crop ET

actual_yield = 0.3
model_yield = outputs_final['Yield (tonne/ha)']

measured_WP = actual_yield/modelled_water_flux.Tr
model_WP = model_yield/modelled_water_flux.Tr

# Regression

WP = pd.DataFrame()
WP['measured'] = measured_WP
WP['model'] = model_WP

fig, ax = plt.subplots(figsize=(8,8)) 
sns.scatterplot(data=WP, x='measured', y='model', s=50, color=".1")
ax.set_title('Rainfed 2014', fontsize=20, weight='bold')
ax.set_xlabel('Measured WP [kg/m$^{3}$]', fontsize=20, weight='bold')
ax.set_ylabel('Modelled WP [kg/m$^{3}$]', fontsize=20, weight='bold')
ax.tick_params(labelsize=18, width='3')
res = stats.linregress(WP.measured, WP.model)
line = res.slope * WP.measured + res.intercept
ax.plot(WP.measured, line, color='C3', linestyle='solid', linewidth='3')
# Draw the 1:1 line through the origin
#min_val = (50, 50) 
max_val = (250, 250)
#ax.plot([min_val, max_val], [min_val, max_val], color='0.0', linestyle='dashed')
ax.axline((0,0), slope = 1, color='green', linestyle='solid')
# Set both x- and y-axis limits
#ax.axis([50, 250, 50, 250])
ax.text(150, 75, f' y = {res.intercept:.2f} + {res.slope:.2f}x', fontsize=20, weight='bold')
plt.tight_layout()

#Student's t-test

"Student's t test for Ho: slope = 1 and offset = 0"

WP_t_statistic_slope = (res.slope - 1)/res.stderr

WP_t_statistic_intercept = res.intercept/res.intercept_stderr

n = len(WP.measured)

WP_p_value_slope = 2 * (1 - stats.t.cdf(np.abs(WP_t_statistic_slope), n - 2))

WP_p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(WP_t_statistic_intercept), n - 2))

print(f'WP_p_value_slope = {WP_p_value_slope}, WP_p_value_intercept = {WP_p_value_intercept}')

# Statistical indicators

R2_WP = r2(measured_WP, model_WP)
RMSE_WP = rmse(measured_WP, model_WP)
NRMSE_WP = nrmse(measured_WP, model_WP)
EF_WP = nse(measured_WP, model_WP)
d_index_WP=agreement_index(measured_WP, model_WP)
PBIAS_WP = pbias(measured_WP, model_WP)
CRM_WP = (np.sum(measured_WP) - np.sum(model_WP))/(np.sum(measured_WP))
MAE_WP = mae(measured_WP, model_WP)
MBE_WP = mean_bias_error(measured_WP, model_WP)

print(f"R2_WP = {R2_WP},RMSE_WP = {RMSE_WP}, NRMSE_WP = {NRMSE_WP}, EF_WP = {EF_WP},
       d-index_WP = {d_index_WP}, PBIAS_WP = {PBIAS_WP}, CRM_WP = {CRM_WP}, MBE_WP = {MBE_WP}")

####################################################################################

plt.show()

'''