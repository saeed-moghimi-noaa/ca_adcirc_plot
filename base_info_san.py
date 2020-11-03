#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
input params

"""
__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2017, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"


from   pynmd.plotting.vars_param import *
from   collections import defaultdict
import datetime


cases = defaultdict(dict)

#### INPUTS ####
#storm_name = 'SAN'
storm_name = 'SANDY'
storm_year = '2012'


# map_plot_options
plot_adc_fort       = True
plot_adc_maxele     = False
plot_nems_fields    = False
plot_forcing_files  = False
plot_nwm_files      = False
local_bias_cor      = True
#HWM proximity limit
prox_max = 0.004



base_dir_sm      = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/noscrub/01_stmp_ca/stmp10_sandy_re/'
hwm_fname        = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/hwm/events/hwm_san.csv'
base_dir_coops_piks = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/ccxxxxxxxx/'
base_dir_obs     = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/coops_ndbc_data/'
nwm_channel_pnts = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/nwm_base_info/channel_points/NWM_v1.1_nc_tools_v1/spatialMetadataFiles/nwm_v1.1_geospatial_data_template_channel_point.nc'
nwm_channel_geom = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/nwm_base_info/channel_geom/nwm_v1.1/nwm_fcst_points_comid_lat_lon-v1-1_ALL.csv'
nwm_results_dir  = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/wrfhydro/test_data/nwm.20180226/'

#ftypes = ['png','pdf']
ftypes = ['png']

san_before_update = False
san_hwrf_update   = True

if True:
    #Base run only tide
    key  = 'atm:n-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a21_SAN_TIDE_v1.1/rt_20180420_h17_m53_s52r550/'      
    cases[key]['label']  = 'Only tide' 
    cases[key]['hsig_file'] = None 
    cases[key]['wdir_file'] = None      
    key0   = key 


if san_before_update:
    #ATM2OCN
    key  = '01-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_SAN_ATM2OCN_v1.1/rt_20180423_h14_m04_s14r624/'        
    cases[key]['label']  = 'ATM2OCN Pre HWRF'  
    cases[key]['hsig_file'] = None 
    cases[key]['wdir_file'] = None    

    #ATM&WAV2OCN 
    wav_inp_dir = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/SANDY/WW3/'
    key  = '02-atm:y-tid:y-wav:y-try01'
    cases[key]['dir']    = base_dir_sm + '/a53_SAN_ATM_WAV2OCN_v1.0/rt_20180423_h14_m09_s08r130/'        
    cases[key]['label']  = 'ATM&WAV2OCN Pre HWRF'  
    cases[key]['hsig_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_hs.nc'
    cases[key]['wdir_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_dir.nc'
    key1 = key

if san_hwrf_update:
    #ATM2OCN
    key  = '03-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a50_SAN_ATM2OCN_v2.1_new_hwrf_land_mask/rt_20190710_h21_m15_s48r723/'        
    cases[key]['label']  = 'ATM2OCN Upd HWRF'  
    cases[key]['hsig_file'] = None 
    cases[key]['wdir_file'] = None  
    #key0 = key

    #ATM&WAV2OCN 
    wav_inp_dir = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/SANDY/WW3/'
    key  = '04-atm:y-tid:y-wav:y-try01'
    cases[key]['dir']    = base_dir_sm + '/a70_SAN_ATM_WAV2OCN_v2.1_new_hwrf_land_mask/rt_20190710_h21_m17_s23r169/'        
    cases[key]['label']  = 'ATM&WAV2OCN Upd HWRF'  
    cases[key]['hsig_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_hs.nc'
    cases[key]['wdir_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_dir.nc'
    key1 = key

#out_dir  = cases[key1]['dir']+'/../01_post_atm2ocn_wav_nwm/'
out_dir  = cases[key1]['dir']+'/../02_post_2019_shachak3/'

#######
defs['elev']['label']  =  'Elev [m]'

dif = False
vec = True

if dif:
    if True:
        defs['elev']['cmap']  =  maps.jetMinWi
        defs['elev']['label']  =  'Surge [m]'
        defs['elev']['vmin']  =  0
        defs['elev']['vmax']  =  5
    else:
        defs['elev']['label']  =  'Wave set-up [m]'
        defs['elev']['cmap']   =  maps.jetWoGn()
        defs['elev']['vmin']  =  -0.5
        defs['elev']['vmax']  =   0.5
else:
    defs['elev']['cmap']  =  maps.jetWoGn()
    defs['elev']['label']  =  'Elev [m]'
    defs['elev']['vmin']  =  -2
    defs['elev']['vmax']  =   2


defs['rad']['vmin']  =  0.0
defs['rad']['vmax']  =  0.01
defs['rad']['cmap']  =  maps.jetMinWi

if False:
    #wind-stress
    defs['wind']['vmin']  =  0.0
    defs['wind']['vmax']  =  0.01
    defs['wind']['label'] = 'Wind force [m$^ \\mathrm{-2}$ s$^ \\mathrm{-2}$] '
else:
    #wind vel
    defs['wind']['vmin']  =  10
    defs['wind']['vmax']  =  30  
    defs['wind']['label'] = 'Wind Speed [m$^ \\mathrm{}$ s$^ \\mathrm{-1}$] '

defs['wind']['cmap']  =  maps.jetMinWi


#added defs
defs['rad' ]['fname'] = 'rads.64.nc'
defs['elev']['fname'] = 'fort.63.nc'
defs['wind']['fname'] = 'fort.74.nc'


varname = 'pmsl'
defs[varname]['vmin']  = 9.2000
defs[varname]['vmax']  = 10.5000
defs[varname]['fname'] = 'fort.73.nc'
defs[varname]['label'] = 'Pressure [mH2O] '
defs[varname]['var']  = 'pressure'


varname = 'hs'
defs[varname]['vmin']  = 1
defs[varname]['vmax']  = 12



track_fname = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/tracks/sandy_bal182012.dat'


#For liverpool and andre report
#key  = 'atm:y-tid:y-wav:n-main-sm24'
#key1 = key
#cases[key]['dir']    = base_dir_sm + '/a52_IKE_ATM2OCN_v1.1/rt_20170801_h20_m24_s42r389/'        
#cases[key]['label']  = 'IKE_HWRF_GFS05d_OC_DA_HSOFS'    
#out_dir0  = base_dir_sm + '/a52_IKE_ATM2OCN_v1.1/'+'01_post/maps/'

#with Gustav
#key  = 'atm:y-tid:y-wav:n-main-sm60'
#cases[key]['dir']    = base_dir_sm + '/a61_IKE_ATM2OCN_v2.0/rt_20170815_h17_m46_s10r529/'
#cases[key]['label']  = 'IKE_HWRF_GFS05d_OC_HSOFS_Gustav'         
#key1 = key
#out_dir0  = base_dir_sm + '/a61_IKE_ATM2OCN_v2.0/'+'01_post/'+cases[key1]['label'] +'/maps/'


#without gustav
#key  = 'atm:y-tid:y-wav:n-main-sm20'
#cases[key]['dir']    = base_dir_sm + '/a52_IKE_ATM2OCN_v1.1/rt_20170801_h20_m15_s00r670//'        
#cases[key]['label']  = 'IKE_HWRF_GFS05d_OC_HSOFS'      
#key1 = key

#############################
#defs['maxele'] = {} 
#defs['maxele'] = defs['elev']
#defs['maxele']['fname'] = 'maxele.63.nc'
#
tim_lim = {}
tim_lim['xmin'] = datetime.datetime(2012, 10, 9, 18, 0) + datetime.timedelta(12.5)
tim_lim['xmax'] = datetime.datetime(2012, 10, 9, 18, 0) + datetime.timedelta(25.125)

if False:
    #plot maps track
    tim_lim['xmin'] = datetime.datetime(2012, 10, 23,1) 
    #plot map area2
    tim_lim['xmin'] = datetime.datetime(2012, 10, 26,23) 
    tim_lim['xmax'] = datetime.datetime(2012, 10, 31,0)

varnames = ['elev']#,'rad'] #,'maxele']#
#varnames = ['rad'] #,'maxele']
#varnames  = ['rad','wind']#,'elev'] #,'maxele']
#varnames = ['elev'] #,'maxele']
#varnames = ['wind']#,'elev'] #,'maxele']
#varnames = ['hs']#,'elev'] #,'maxele']
#varnames = ['elev'] #,'maxele']
#
regions = ['san_area2','san_area','san_newyork','san_track','san_delaware','san_jamaica_bay']
#regions = ['hsofs_region','san_track','san_area2','san_delaware','san_area']
#regions = ['san_jamaica_bay']
#regions = ['san_delaware']#,'san_area2']
#regions = ['san_area']  #anim
#regions = ['san_area']  # HWM
regions = ['san_newyork']

####################
#tim_lim['xmin'] = datetime.datetime(2017, 9, 5,23)
#tim_lim['xmax'] = datetime.datetime(2017, 9, 7,12)
#regions = ['burbuda_zoom' ,'puertorico_shore'] #,'carib_irma']
#####################
#tim_lim['xmin'] = datetime.datetime(2017, 9, 7,11)
#tim_lim['xmax'] = datetime.datetime(2017, 9, 9,12)

#tim_lim['xmin'] = datetime.datetime(2017, 9, 9,10)
#tim_lim['xmax'] = datetime.datetime(2017, 9, 12,12)
#regions = ['cuba_zoom','key_west_zoom']
#####################

#regions = ['ike_region']
#
#tim_lim['xmin'] = datetime.datetime(2017, 9, 6 )
#tim_lim['xmax'] = datetime.datetime(2017, 9, 12 )

#

    
#latp = 29.2038
#lonp = -92.2285
#i,prox = find_nearest1d(xvec = lon,yvec = lat,xp = lonp,yp = latp)

# for COOPS time series plot
station_selected_list = None
   
    
    
"""


if False:
    #ATM2OCN
    key  = '01-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_SAN_ATM2OCN_v2.0/rt_20180510_h12_m53_s33r830/'        
    cases[key]['label']  = '3DVar'         
    #key0   = key 

    key  = '03-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_SAN_ATM2OCN_v2.0/rt_20180510_h13_m02_s21r175/'        
    cases[key]['label']  = 'Hybrid'                
                   
    key  = '05-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_SAN_ATM2OCN_v2.0/rt_20180510_h13_m06_s43r786/'        
    cases[key]['label']  = 'Operational'                
    #key1 = key

    key  = '07-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_SAN_ATM2OCN_v2.0/rt_20180510_h12_m57_s57r209/'        
    cases[key]['label']  = 'ENS_ch75'                
    #key1 = key

    #ATM&WAV2OCN 
    wav_inp_dir = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/SANDY/WW3/'
    key  = '02-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_SAN_ATM_WAV2OCN_v2.0/rt_20180510_h13_m15_s13r719/'        
    cases[key]['label']  = '3DVar WAV'         
    cases[key]['hsig_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_hs.nc'
    cases[key]['wdir_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_dir.nc'
    key1 = key
     
    key  = '04-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_SAN_ATM_WAV2OCN_v2.0/rt_20180510_h13_m23_s44r792/'        
    cases[key]['label']  = 'Hybrid WAV'                
    cases[key]['hsig_file'] = wav_inp_dir + 'ww3.HWRF.Hybrid.2012_hs.nc'
    cases[key]['wdir_file'] = wav_inp_dir + 'ww3.HWRF.Hybrid.2012_dir.nc'
                   
    key  = '06-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_SAN_ATM_WAV2OCN_v2.0/rt_20180510_h13_m28_s11r309/'        
    cases[key]['label']  = 'Operational WAV'                
    cases[key]['hsig_file'] = wav_inp_dir + 'ww3.HWRF.Operational.2012_hs.nc'
    cases[key]['wdir_file'] = wav_inp_dir + 'ww3.HWRF.Operational.2012_dir.nc'
    
    key  = '08-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_SAN_ATM_WAV2OCN_v2.0/rt_20180510_h13_m19_s27r209/'        
    cases[key]['label']  = 'ENS_ch75 WAV'                
    cases[key]['hsig_file'] = wav_inp_dir + 'ww3.HWRF.ENS_CH75.2012_hs.nc'
    cases[key]['wdir_file'] = wav_inp_dir + 'ww3.HWRF.ENS_CH75.2012_dir.nc'





if False:
    #ATM&WAV2OCN 
    wav_inp_dir = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/SANDY/WW3/'
    key  = '02-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_SAN_ATM_WAV2OCN_v2.0/rt_20180510_h13_m15_s13r719/'        
    cases[key]['label']  = '3DVar WAV'         
    cases[key]['hsig_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_hs.nc'
    cases[key]['wdir_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_dir.nc'
    key1 = key
     


"""