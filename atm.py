
"""Collection of atmospheric physics methods - Python 3

.. todo:: Update documentation
.. todo:: Add tested footprint functions

"""

from __future__ import division

import numpy as np
import pandas as pd

def air_mr2(t, RH, press = 101325):
    """Mixing ratio - t in C """
    es = 6.11*np.exp((17.6*t)/(t+243))*100    
    e= RH*es/100                    # Pa
    w = (0.622 * e)/ (press - e)    # kg/kg
    return w*1000

def air_mr(t, RH, press = 101325):
    """Mixing ratio - t in C """
    es = 6.11*np.exp((17.6*t)/(t+243))*100    
    e= RH*es/100                    # Pa
    w = (0.622 * e)/ (press - e)    # kg/kg
    return w*1000

def es2(T):
    """Tetens formula, es (kPa), T in Celcius"""
    return 0.6108*np.exp((17.27*T)/(T+237.3))


def calc_vpd(airt, rh):
    """Air T in Celsius, rh in %

    Output is in kPa"""

    es = es2(airt) # kPa
    vpd = ((100.0 - rh) / 100.0) * es
    return vpd
    


def sat_spe_hum(T, p=101325.0):
    # Saturation specific humidity
    # Output in g/kg
    p=p/100.0
    ew = 6.1121*(1.0007+3.46e-6*p)*np.exp((17.502*T)/(240.97+T)) # in mb
    q  = 0.62197*(ew/(p-0.378*ew))                         # mb -> kg/kg
    return q*1000

def calc_spe_hum(rh, T, p=101325.0):
    """Calculate specific humidity at T(C) and RH(%)
    
    qair in kg/kg units!!!
    """
    es = 6.11*np.exp((17.6*T)/(T+243))
    e = rh/100. * es
    p_mb = p/100.
    qair = (0.622 * e) / (p_mb - (0.378 * e))
    return qair


def hsieh00(zm, z0, L):
    """Analytic footprint peak using Hsieh 2000

    zm: measurement height
    z0: roughness length
    L: Obukhov length"""
    # zm = 9 - d                  # measurement height
    # z0 = 0.1                   # Roughness length
    zu = zm*(np.log(zm/z0) - 1 + z0/zm)
    zL = zu / L
    vk = 0.41                   # von karman

    DD = 0.97
    PP = 1.0
    if (abs(zL) < 0.04):
        DD = 0.97
        PP = 1.0
    elif (zL < 0):
        DD = 0.28
        PP = 0.59
    elif zL > 0:
        DD = 2.44
        PP = 1.33

    # Peak distance eq.19
    x = (DD * zu**PP * abs(L)**(1-PP)) / (2*vk**2)

    return x

def calc_monthly_stats(df, df_rain, t_name, rh_name):
    """ Calculate T,RH and rain monthly values
    """
    mm_min_f = []
    mm_max_f = []
    mm_avg_f = []
    rh_min_f = []
    rh_max_f = []
    rh_avg_f = []

    years = pd.Series(df.index.year.tolist()).unique()
    year_col = []
    month_col = []

    for yy in years:
        months = pd.Series(df[df.index.year == yy].
                           index.month.tolist()).unique()
        for i in months:
            mm = df[(df.index.month == i) & (df.index.year == yy)]
            mm_min = []
            mm_max = []
            mm_avg = []
            rh_min = []
            rh_max = []
            rh_avg = []


            for dd in np.unique(mm.index.day):
                day = mm[mm.index.day == dd]
                mm_max.append(day[t_name].max())
                mm_min.append(day[t_name].min())
                mm_avg.append(day[t_name].mean())
                rh_max.append(day[rh_name].max())
                rh_min.append(day[rh_name].min())
                rh_avg.append(day[rh_name].mean())

            mm_min_f.append(round(np.nanmean(mm_min), 2))
            mm_max_f.append(round(np.nanmean(mm_max), 2))
            mm_avg_f.append(round(np.nanmean(mm_avg), 2))
            rh_min_f.append(round(np.nanmean(rh_min), 2))
            rh_max_f.append(round(np.nanmean(rh_max), 2))
            rh_avg_f.append(round(np.nanmean(rh_avg), 2))

        month_col = month_col+list(months)
        year_col = year_col+list(np.ones(months.shape[0])*yy)
    year_col = [int(i) for i in year_col]
    
    # Construct the year and month column into new data frame
    #date_str = map(lambda x, y : str(x)+str(y), year_col, month_col)
    date_str = [(str(y)+'-'+str(month_col[x])) for x,y in enumerate(year_col)] 
    date_obj = pd.to_datetime(date_str, format="%Y-%m")
    data = pd.DataFrame(index=date_obj)

    
    data["T_avg"] = mm_avg_f
    data["T_daily_min_avg"] = mm_min_f
    data["T_daily_max_avg"] = mm_max_f

    data["RH_avg"] = rh_avg_f
    data["RH_daily_min_avg"] = rh_min_f
    data["RH_daily_max_avg"] = rh_max_f

    data["rain_sum_mm"] = df_rain.resample('M', how=np.nansum).values

    return data


def calc_inc_lwrad(temp, rh):
    """Simple approximation of incoming longwave radiation without cloud fraction

    temp -- air T in Celsius
    rh -- relative humidity %
    output -- lwrad in W/m2

    On the development of a simple downwelling
    longwave radiation scheme - Sridhar (2002)

    TODO: Function with cloud cover effects would be more accurate
    """

    def es(T):
        # Saturation vapour pressure.
        #  T is the air temperature, C
        #  es is the saturation vapour pressure in kPa
        es = 0.6106 * np.exp(17.27 * T / (T + 237.3))
        return es


    tempK = temp+273.15

    coeff = 1.31 # original value
    sb = 5.6704E-8 # Stefan-Boltzman constant, W/m2/K4
    es = es(temp) # kPa
    e = rh/100.0*es
    lwrad = coeff*(10*es/tempK)**(1.0/7.0)*sb*tempK**4

    return lwrad

def rh_to_qair(rh, T, press = 1013.250):
    """Convert RH to specific humidity [kg/kg]

    Parameters
    ----------
    rh [%]
    T [C]
    press [mb]

    Example
    -------
    rh_to_qair(36, 14, 857) = 0.004184 [kg/kg]
    """
    
    rh = rh/100
    Tc = T 
    es = 6.112 * np.exp((17.67 * Tc) / (Tc +243.5))
    e = rh * es
    p_mb = press
    qair = (0.622 * e) / (p_mb - (0.378 * e))
    return qair


def vapor_pressure(T):
    """
    Estimates the water vapor pressure (kPa) at a given temperature (C).
    """
    return 0.6108 * np.exp((17.27 * T) / (T + 237.3))

@np.vectorize
def et_penman(u2, P, Pv, temp, nrad, soilf, daytime, timestep=3600):
    """ASCE Standardized Reference Evapotranspiration Equation

    Input data must be per hour
    Parameters are for short reference and daytime only!

    u2 = wind at 2m [m/s]
    P = Pressure [kPa]
    Pv = VPD [kPa]
    temp = Air temp [C]
    nrad = net_rad [W/m2]
    soilf = G [W/m2]    
    daytime = True when it is daytime
    timestep = data timestep in seconds = 3600
    """

    # TODO: Input sanity checks!!!!!
    
    # check and replace dewpoints lower than tmin (physically impossible)

    # dewpoint = numpy.minimum(data['temperature'], data['dewpoint'])

    netrad =  nrad * timestep / 10**6 # (MJ/m2/hour)
    soilf = soilf * timestep / 10**6 # (MJ/m2/hour)
    # sat vap pressure
    Ps = vapor_pressure(temp)

    # estimate the vapor pressure curve slope (kPa C-1) at mean temperature
    d = (4098 * Ps) / (temp + 237.3)**2

    T = temp + 273.15

    # numerator constant that changes with reference type and calculation
    # time step (K mm s3 Mg-1 d-1 or K mm s3 Mg-1 h-1)
    Cn = 37

    if daytime:
        Cd = 0.24
    else:
        Cd = 0.96
    

    # estimate the psychrometric constant (kPa C-1)
    # equation is gamma = cp(T) * P * MWair / latent_heat / MWwater
    # this uses cp for water at T = 20 C
    g = 0.000665 * P
    
    ET = ((0.408 * (netrad - soilf) * d + Cn * u2 / T * (Ps - Pv) * g) /
          (d + (g * (1 + Cd * u2))))

    return ET

def calc_lv(airt):
    """Calculate latent heat of vaporization

    Keyword Arguments:
    airt -- [C]

    Returns:
    Lv -- [J/kg]
    """
    T = 273.15+airt    
    return (3147.5 - 2.37*T)*1000

def latent_mm(ser, airt, seconds):
    """Convert latent heat flux to mm

    Keyword Arguments:
    ser     -- LE
    airt    -- Air temperature
    seconds -- how many seconds? 1800
    """
    Lv = calc_lv(airt)
    return ser/Lv*seconds
    

def calc_es(T):
    """Tetens formula, es (kPa), T in Celcius"""
    return 0.6108 * np.exp((17.27 * T) / (T + 237.3))


def calc_delta(airt):
    """Calculate the slope of temperature vs. vpd curve

    airt [C]
    s = slope [Pa/C]
    """
    es = calc_es(airt)  # kPa
    delta = es * 4098.0 / ((airt + 237.3)**2) * 1000
    return delta

def calc_delta2(airt):
    '''Calculate the slope of the temperature - vapour pressure curve

    - airt: air temperature [Celsius].
    - Delta: (array of) slope of saturated vapour curve [Pa K-1].
    '''
    # calculate saturation vapour pressure at temperature
    es = es2(airt)*1000 # in Pa
    # Convert es (Pa) to kPa
    es = es / 1000.0
    # Calculate Delta
    Delta = es * 4098.0 / ((airt + 237.3)**2) * 1000
    return Delta # in Pa/K


def calc_Eeq(airt,Rn, G):
    """Calculate equilibrium evaporation

    """
    delta = calc_delta2(airt)# Pa/K
    gamma = 66.5 # Pa/K
    # L = 2.5e6
    # epsilon = 1800

    eq = (delta)/(delta+gamma)*(Rn-G)

    return eq
def calc_PET(leq):
    """Calc PET from eq evap ()

    ASSUMING hourly data!
    leq
    mean        66
    std        104
    max        505
    """
    return (1.26 * leq / 29. /24.) # 0 to 1 mm



def calc_pt_soil_evap(airt, Rn, G, soil_frac):
    
    """Calculate soil evaporation from Priestley Taylor

    # NOTICE: gamma is constant! Could be a function!

    Bagayoko, F., Yonkeu, S., Elbers, J. and van de Giesen, N.: Energy
    partitioning over the West African savanna: Multi-year evaporation and
    surface conductance measurements in Eastern Burkina Faso, Journal of
    Hydrology.

    Keyword Arguments:
    airt      -- [C]
    Rn        -- [W/m2]
    G         -- [W/m2]
    soil_frac -- %

    Output:
    evap -- W/m2
    """
    delta = calc_delta(airt) # [Pa/C]
    gamma = 0.66  # Pa/K
    
    evap = soil_frac*(delta/(delta+gamma)*(Rn-G))
    
    return evap



def calc_interpulse(t, x):
    x = x - x.mean()
    n = x.shape[0]
    u1 = x[0:n - 1]
    u2 = x[1:n]
    pr = u1 * u2
    # np.nonzero(a>0.5)
    Dur = t[np.where(pr < 0)]
    return np.diff(Dur)

def calc_storm_depth_freq(dat):
    """Calculate the two Prepitation characteristics - Ecohydrology

    # Has been tested to work!

    # NOTICE: Edge issue
    The interpulse can make the last interval 2 days less.
    No difference for long times series

    INPUT
    df.precip

    alpha and lambda
    """
    p = dat.resample("D").sum()
    alpha = p[p> 0.0].mean() # mm/day
    
    # print("alpha =  {:.2f} mm/day".format(p[p> 0.0].mean()))

    ta = p.copy()
    ta[ta>0.0] = 1.0
    time = np.arange(0,ta.shape[0])
    ip_ns = calc_interpulse(time, ta.values)
    lam = 1/np.mean(ip_ns)
    # print("lambda =  {:.2f} storms/day".format(lam))
    # return alpha, lam, ta, ip_ns
    return alpha, lam

def calc_metrics(mm, var, fmean=False):
    """Calculate annual precipitation metrics

    """

    if fmean:
        Rk= mm.groupby('ycol')[var].mean()
    else:
        Rk= mm.groupby('ycol')[var].sum()

    # Monthly prob distribution
    mm['pk'] = np.nan
    for m in mm.iterrows():    
        mm.loc[m[0], 'pk'] = m[1][var]/Rk.loc[m[1].ycol]

    # Relative entropy for each hydrological year!
    Dk = mm.groupby('ycol').apply(lambda x: np.sum(x.pk*np.log2(x.pk/(1/12.))))

    # Seasonality index
    Sk = Dk * Rk/Rk.max()

    # # Centroid
    # Ck = 1/Rk*mm.groupby('ycol').apply(lambda x: np.sum(np.arange(1,13)*x[var]))

    # mm['Ck'] = np.nan
    # for m in mm.iterrows():    
    #     mm.loc[m[0], 'Ck'] = Ck.loc[m[1].ycol]

    # zk = []
    # for y in mm.ycol.unique():
    #     tmp = 1/(Rk.loc[y])*np.sum((np.arange(1,13)-Ck.loc[y])**2*mm[mm.ycol == y][var])
    #     zk.append(np.sqrt(tmp))

    out = pd.DataFrame(Dk)
    out.columns = ['dk']
    out['sk'] = Sk
    return out
