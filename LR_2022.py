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

met = pd.read_csv('Maktau_AWS_1h.dat')

met = met.set_index(pd.to_datetime(met['Unnamed: 0']))
met.index.name = ''

smplot = pd.read_csv("Maktau5.csv")

smplot = smplot.set_index(pd.to_datetime(smplot.Measurement_Time))

# Incoming global radiation scaled to attain Net radiation

met['net_mod'] = 0.78*met.SlrW_Avg
met['vpd'] = atm.calc_vpd(met.AirTC_Avg, met.RH)
met['soilf'] = met.filter(regex='Flux_corr').mean(axis=1)
met['daytime'] = met.SlrW_Avg > 10
met['Rain_mm_Tot_corr'] = 1.07*met.Rain_mm_Tot

met['et_pm'] = atm.et_penman(met.WS_ms_S_WVT, 90, met.vpd, met.AirTC_Avg, met.net_mod, met['soilf'], met.daytime, timestep=3600)

per = slice('20220101','20230101')

dat = met[per]

dd = dat.resample("D").mean()

dd['Precipitation'] = dat.resample("D").Rain_mm_Tot_corr.sum()
dd['ReferenceET'] = dat.resample("D").et_pm.sum()
dd['MinTemp'] = dat.resample("D").AirTC_Avg.min()
dd['MaxTemp'] = dat.resample("D").AirTC_Avg.max()
dd['Date'] = dd.index.values
#dd['evap_corr_mm'] = dat.resample("D").evap_corr_mm.sum()

#print(dd.ndvi_maize) #,red_bush,nir_bush,ndvi_bush


# Model input

wdf = dd[['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET' , 'Date']]

wdf = wdf.reset_index()

wdf = wdf.drop([''], axis=1)

#######################
# Aquacrop simulation #
#######################

sim_start = '2022/03/17'

sim_end = '2022/08/13'

custom = Soil('custom', dz=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

custom.add_layer(thickness=0.1, thWP=0.06, thFC=0.17, thS=0.45, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.1, thWP=0.08, thFC=0.20, thS=0.42, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.1, thWP=0.08, thFC=0.22, thS=0.42, Ksat=255, penetrability=100)

custom.add_layer(thickness=0.2, thWP=0.10, thFC=0.23, thS=0.42, Ksat=1000, penetrability=100)

custom.add_layer(thickness=0.5, thWP=0.10, thFC=0.23, thS=0.42, Ksat=1000, penetrability=100)

# Initial water content

initWC = InitialWaterContent(wc_type='Num', method='Layer', depth_layer=[1,2,3,4,5], value=[0.04,0.06,0.08,0.09,0.09])

#initWC = InitialWaterContent(wc_type='Prop', method='Layer', depth_layer=[1,2,3,4,5], value=['FC','FC','FC','FC','FC'])

#initWC = InitialWaterContent(wc_type='Prop', method='Layer', depth_layer=[1,2,3,4,5], value=['WP','WP','WP','WP','WP'])


period = slice('20220317','20220813')

# AWS soil moisture

dd_measured = dd[period]

measured_met = dd_measured[['VWC_10cm_corr', 'VWC_30cm_corr', 'VWC_50cm_corr', 'ndvi_maize']]

# Soil moisture in the plot

sm_dd = smplot.resample("D").mean()[period]

print(sm_dd)

print(measured_met)


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

out = outputs_water_storage[0].iloc[:-100,:]

#print(out)
out['swc10'] = measured_met.VWC_30cm_corr.iloc[:-100].values
out['swc30'] = measured_met.VWC_30cm_corr.iloc[:-100].values
out['swc50'] = measured_met.VWC_50cm_corr.iloc[:-100].values

out['ET_mod'] = modelled_water_flux.modelled_ET.iloc[:-100]
#out['ET_measured'] = measured_met.evap_corr_mm.iloc[:-2].values
out['ET_PM'] = dd['ReferenceET'][period].iloc[:-100].values
out['Precipitation'] = dd['Precipitation'][period].iloc[:-100].values
out['ndvi_maize'] = dd['ndvi_maize'][period].iloc[:-100].values

print(out.ndvi_maize)

measured_canopy = ((105.427*out.ndvi_maize) + (-6.501))/100



# P1: 10 cm, P2: 20 cm

#out['swc10_plot'] = sm_dd.Port_1.iloc[:-100].values

#out['swc20_plot'] = sm_dd.Port_2.iloc[:-100].values


fig, ax = plt.subplots(2, figsize=(8,8))
out.ET_mod.plot(ax=ax[0])
#out.ET_measured.plot(ax=ax[0])
out.ET_PM.plot(ax=ax[0])
ax[0].legend()

fig1, ax = plt.subplots(2,1, figsize=(8,8))
out.th1.plot(ax=ax[0])
out.swc10.plot(ax=ax[0])
#out['swc10_plot'].plot(ax=ax[0])
#out.swc20_plot.plot(ax=ax[0])
out.swc30.plot(ax=ax[0])
out.swc50.plot(ax=ax[0])
out.Precipitation.plot(ax=ax[1])
ax[0].legend()

#
# Plot sm 20cm
fig2, ax = plt.subplots(1, figsize=(8,8))
out.th2.plot(ax=ax)
#out.swc20_plot.plot(ax=ax)

#
fig3, ax = plt.subplots(1,1, figsize=(8,8)) 
out.th3.plot(ax=ax)
out.swc30.plot(ax=ax)

# 
fig4, ax = plt.subplots(1,1, figsize=(8,8)) 
out.th5.plot(ax=ax)
out.swc50.plot(ax=ax)
    
# Plot canopy cover
fig, ax = plt.subplots(1, figsize=(8,8))
out['model_canopy_cover'] = modelled_crop_growth.canopy_cover.iloc[:-100].values
out.model_canopy_cover.plot(ax=ax, fontsize=18)
measured_canopy.plot(ax=ax)
#out.modis_CC_linear_cereals.plot(ax=ax)
ax.legend(fontsize=16)

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

print(model_tsw)

swc = [out.swc10, out.swc30, out.swc50]

z = [100, 300, 500]

measured_tsw = calculate_sws(swc, z)

print(measured_tsw)

R2 = r2(measured_tsw, model_tsw)
RMSE = rmse(measured_tsw, model_tsw)
NRMSE = nrmse(measured_tsw, model_tsw)
result=agreement_index(measured_tsw, model_tsw)
EF = nse(measured_tsw, model_tsw)
PBIAS = pbias(measured_tsw, model_tsw)
CRM = (np.sum(measured_tsw) - np.sum(model_tsw))/(np.sum(measured_tsw))

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

#predictions = out.th5
#targets = out.swc50

#def nse(predictions, targets):
#    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

#result2 = nse(predictions, targets)

print(f"R2 = {R2},RMSE = {RMSE}, NRMSE = {NRMSE}, EF = {EF}, d-index = {result}, PBIAS = {PBIAS}, CRM = {CRM}")

print(f"R2_30cm = {R2_30cm},RMSE_30cm = {RMSE_30cm}, NRMSE_30cm = {NRMSE_30cm}, EF_30cm = {EF_30cm}, d-index_30cm = {result_30cm}, PBIAS_30cm = {PBIAS_30cm}")

print(f"R2_50cm = {R2_50cm},RMSE_50cm = {RMSE_50cm}, NRMSE_50cm = {NRMSE_50cm}, EF_50cm = {EF_50cm}, d-index_50cm = {result_50cm}, PBIAS_50cm = {PBIAS_50cm}")


def perform_ttest(group1, group2, alpha=0.05):
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    
    # Calculate degrees of freedom
    df = len(group1) + len(group2) - 2
    
    # Determine if the result is significant
    is_significant = p_value < alpha
    
    # Print results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
    print(f"Degrees of freedom: {df}")
    print(f"Significant: {'Yes' if is_significant else 'No'}")
    
    return t_statistic, p_value, df, is_significant

group1 = measured_tsw
group2 = model_tsw

perform_ttest(group1, group2)

fig, ax = plt.subplots(figsize=(10,7))

measured_tsw.plot(ax=ax, label = 'Measured SWC', color='r', linewidth='3')
model_tsw.plot(ax=ax, label = 'Modelled SWC', color='b', linewidth='3')
#ax.axhline(y=0, color='r', linestyle='-')
#ax.axhline(y=5, color='r', linestyle='-')

# Set the y-axis limits (optional)
#ax.set_xlim(-7, 124)
#ax.set_ylim(60, 250)

# Add labels and title (optional)
ax.tick_params(labelsize=18, width='3')
ax.set_xlabel('DAS', fontsize=20, weight='bold')
ax.set_ylabel('SWC (mm)', fontsize=20, weight='bold')
ax.set_title('Rainfed 2022', fontsize=20, weight='bold')
ax.legend(loc='best', fontsize=16)

####################################################################

fig, ax = plt.subplots(1, figsize=(10,7))
out.Precipitation.plot(ax=ax, color='r', linewidth='3')
ax.tick_params(labelsize=18, width='3')
ax.set_ylabel('Precipitation (mm)', fontsize=20, weight='bold')
ax.set_xlabel('DAS', fontsize=20, weight='bold')
ax.legend(loc='upper left', fontsize=16)
ax=ax.twinx()
measured_tsw.plot(ax=ax, color='b', linewidth='3', label='measured swc')
model_tsw.plot(ax=ax, color='green', linewidth='3', label='simulated swc')
ax.tick_params(labelsize=18, width='3')
ax.set_ylabel('SWC (mm)', fontsize=20, weight='bold')
ax.set_title('Rainfed 2022: Precipiation vs Soil water content', fontsize=20, weight='bold')
ax.legend(loc='upper right', fontsize=16)

# Add labels and title (optional)


# Display the plot
plt.grid(True)
plt.tight_layout()
#print(plt.style.available)
#plt.style.use('style')

print(f'Total rainfall is {np.sum(dd.Precipitation[period])} mm')

plt.show()