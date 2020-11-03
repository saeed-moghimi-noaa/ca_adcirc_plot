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
base_dir_sm   = '/scratch2/COASTAL/coastal/noscrub/Saeed.Moghimi/01_stmp_ca/stmp20_florence/'
track_fname   = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/tracks/florence_bal062018.dat'
hwm_fname     = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/hwm/events/hwm_ike.csv'
base_dir_coops_piks = 'xxx'
base_dir_obs        = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/coops_ndbc_data/'

nwm_channel_pnts    = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/nwm_base_info/channel_points/NWM_v1.1_nc_tools_v1/spatialMetadataFiles/nwm_v1.1_geospatial_data_template_channel_point.nc'
nwm_channel_geom    = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/nwm_base_info/channel_geom/nwm_v1.1/nwm_fcst_points_comid_lat_lon-v1-1_ALL.csv'
nwm_results_dir     = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/wrfhydro/test_data/nwm.20180226/'



name = 'FLORENCE'
year = '2018'

# map_plot_options
plot_adc_fort       = True
plot_adc_maxele     = False
plot_nems_fields    = False
plot_forcing_files  = True
plot_nwm_files      = False
plot_transects      = False


vec = False
local_bias_cor = True
#HWM proximity limit
prox_max = 0.0075 * 1  #grid size * n

if False:
    #Base run only tide
    key  = '00-atm:n-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a10_FLO_OCN_SPINUP_v1.0/rt_20200306_h20_m10_s48r471/' 
    cases[key]['label']  = 'Only tide' 
    cases[key]['hsig_file'] = None
    cases[key]['wdir_file'] = None
    key0 = key 

if True:
    # 
    key  = '09-atm:y-tid:y-wav:n'
    #cases[key]['dir']    = base_dir_sm + '/a50_FLO_ATM2OCN_v2.1_new_hwrf_land_mask/rt_20200306_h20_m42_s52r091/'   
    cases[key]['dir']    = base_dir_sm + '/a50_FLO_ATM2OCN_v2.2_extended/rt_20200313_h18_m12_s49r072/'   
    cases[key]['label']  = 'ATM'    
    cases[key]['hsig_file'] = None
    cases[key]['wdir_file'] = None
    key0 = key 

if True:
    #2way wav-ocn
    key  = '11-atm:y-tid:y-wav:y'
    #cases[key]['dir']    = base_dir_sm + 'a70_FLO_ATM_WAV2OCN_v2.1_new_hwrf_land_mask_hera/rt_20200306_h21_m45_s06r726/' 
    cases[key]['dir']    = base_dir_sm + 'a70_FLO_ATM_WAV2OCN_v2.2_extended/rt_20200313_h18_m16_s45r503/'      
    cases[key]['label']  = 'ATM&WAV'    
    cases[key]['hsig_file'] = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/nsemodel_inps/hsofs_forcings/flo_v0/inp_ww3/ww3.Florence.HWRF_HRRR_CFSR.201809_hs.nc'
    cases[key]['wdir_file'] = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/nsemodel_inps/hsofs_forcings/flo_v0/inp_ww3/ww3.Florence.HWRF_HRRR_CFSR.201809_dir.nc'
    key1 = key 

    ###########

#out_dir  = cases[key1]['dir'] +'/../05_post_coupled/'
out_dir  = base_dir_sm  + '/06_post_coupled/'

#########################  Variable Limits Settings ####################################
tim_lim = {}
#tim_lim['xmin'] = datetime.datetime(2008, 9,  5 )
#tim_lim['xmax'] = datetime.datetime(2008, 9, 13,11 )

#tim_lim['xmin'] = datetime.datetime(2008, 9, 13,10 ) - datetime.timedelta(4)
#tim_lim['xmax'] = datetime.datetime(2008, 9, 13,10 ) + datetime.timedelta(1)

#tim_lim['xmin'] = datetime.datetime(2008, 9, 13,10 ) - datetime.timedelta(0.5)
#tim_lim['xmax'] = datetime.datetime(2008, 9, 13,10 ) + datetime.timedelta(0.5)

#tim_lim['xmin'] = datetime.datetime(2008, 9, 7,15 ) 
#tim_lim['xmax'] = datetime.datetime(2008, 9, 9,5 )

#maps anim
#tim_lim['xmin'] = datetime.datetime(2008, 9,  5 )
tim_lim['xmin'] = datetime.datetime(2018, 9, 10)  -  datetime.timedelta(20.0)
tim_lim['xmax'] = datetime.datetime(2018, 9, 18)  +  datetime.timedelta(20.0)

if True:
    defs['elev']['cmap']  =  maps.jetMinWi
    defs['elev']['label']  =  'Surge [m]'
    defs['elev']['vmin']  =  0
    defs['elev']['vmax']  =  7
    
    #defs['elev']['vmin']  =  2
    #defs['elev']['vmax']  =  7    
    
else:
    defs['elev']['label']  =  'Wave set-up [m]'
    defs['elev']['cmap']   =  maps.jetWoGn()
    defs['elev']['vmin']  =  -0.5
    defs['elev']['vmax']  =   0.5

#rad-stress
defs['rad']['vmin']  =  0.0
defs['rad']['vmax']  =  0.01
defs['rad']['cmap']  =  maps.jetMinWi

if True:
    #wind-stress
    defs['wind']['vmin']  =  0.0
    defs['wind']['vmax']  =  0.01
    defs['wind']['label'] = 'Wind force [m$^ \\mathrm{-2}$ s$^ \\mathrm{-2}$] '
else:
    #wind vel
    defs['wind']['vmin']  =  10
    defs['wind']['vmax']  =  40  
    defs['wind']['label'] = 'Wind Speed [m$^ \\mathrm{}$ s$^ \\mathrm{-1}$] '

defs['wind']['cmap']  =  maps.jetMinWi

################################################
varname = 'pmsl'
defs['pmsl']['fname']   = 'fort.73.nc'
#defs['pmsl']['label']   = 'Pressure [mH2O] '
defs[varname]['label']  = 'pressure [Pa]'
defs['pmsl']['var']     = 'pressure'
if True:
    defs['pmsl']['vmin']  = 9.2000
    defs['pmsl']['vmax']  = 10.5000
else:
    defs['pmsl']['vmin']  =  92000
    defs['pmsl']['vmax']  =  101000
    #defs[varname]['vmin']  =  val.min() // 1000 * 1000

defs['hs']['vmin']  =  0
defs['hs']['vmax']  =  8                    ### plot stuff
#defs['hs']['vmax']  =  val.max() // 5 * 5                    ### plot stuff

#added defs
defs['rad' ]['fname'] = 'rads.64.nc'
defs['elev']['fname'] = 'fort.63.nc'
defs['wind']['fname'] = 'fort.74.nc'

##### SELECLT VARNAME TO PLOT ########################
#varnames = ['elev','rad','wind'] #,'maxele']#
#varnames = ['rad'] #,'maxele']
#varnames  = ['rad','wind']#,'elev'] #,'maxele']
#varnames = ['elev'] #,'maxele']
#varnames = ['hs','wind','elev'] #,'maxele']
varnames = ['elev'] #,'maxele']
#varnames = ['elev'] #,'maxele']
varnames = ['wind']#,'elev'] #,'maxele']



##### SELECLT GEO REGION TO PLOT ########################
#regions = ['isa_landfall_zoom','isa_landfall','isa_local','isa_region','hsofs_region']
regions = ['hsofs_region']

#station_selected_list = None     # for COOPS time series plot


station_selected_list = None 

"""

station_selected_list =  [
    '8771013',
    '8771510',
    '8770570',
    '8771450',
    '8768094',
    '8770933',
    '8770971',
    '8770777',
    '8770475',
    '8764044',
    #'8764227',
    #'8760417',
    ]




station_selected_list =  [
    '8771013',
    '8771510',
    '8770570',
    '8768094',
    #'8764044',
    #'8764227',
    #'8760417',
    #wave effects
    ]






#   
    




if False:
    key  = '10-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a60_IKE_fully_coupled/rt_no_rad_stress/'        
    cases[key]['label']  = 'GFS05d_OC_DA_Wav'    
    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/noscrub/Saeed.Moghimi/stmp5/a60_IKE_fully_coupled/rt_no_rad_stress/RUN1/ww3.run1.ike.2008_hs.nc'
    cases[key]['wdir_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/EC2001V516WindZeta/V418/test2WindLevel/ww3.WindLevel20080910_dir.nc'
    ###########
    ###########
    key  = '11-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a60_IKE_fully_coupled/rt_fully_coupled/'        
    cases[key]['label']  = 'GFS05d_OC_DA_Fully_coupled'    
    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/noscrub/Saeed.Moghimi/stmp5/a60_IKE_fully_coupled/rt_fully_coupled/RUN2/ww3.run2.ike.2008_hs.nc'
    cases[key]['wdir_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/EC2001V516WindZeta/V418/test2WindLevel/ww3.WindLevel20080910_dir.nc'
    key1 = key

     
if False:
    key  = '10-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_IKE_ATM_WAV2OCN_v3.0/rt_20170926_h18_m14_s03r034/'        
    cases[key]['label']  = 'GFS05d_OC_DA_Wav'    
    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/EC2001V516WindZeta/V418/test2WindLevel/ww3.WindLevel20080910_hs.nc'
    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/Ike/WW3/StandAlone/GFS05d_OC_DA/ww3.WND.GFS05dOcDa.200809_hs.nc'
    cases[key]['wdir_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/EC2001V516WindZeta/V418/test2WindLevel/ww3.WindLevel20080910_dir.nc'
    ###########
    ###########
    key  = '11-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a60_IKE_fully_coupled/rt_cfsr@20080903_2day_sbys_ww3@T188/'        
    cases[key]['label']  = 'GFS05d_OC_DA_Fully_coupled'    
    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/COUPLED/DEC13/transfer/ww3.run1.ike.2008_hs.nc'
    cases[key]['wdir_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/EC2001V516WindZeta/V418/test2WindLevel/ww3.WindLevel20080910_dir.nc'
    key1 = key






if False:
    key  = '01-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_IKE_ATM2OCN_v3.0/rt_20170926_h18_m02_s21r086/'        
    cases[key]['label']  = 'GFS05d_OC'         
    key0 = key 

if False:
    key  = '03-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_IKE_ATM2OCN_v3.0/rt_20170926_h17_m57_s23r197/'        
    cases[key]['label']  = 'GFS05d'                
                   
    key  = '05-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_IKE_ATM2OCN_v3.0/rt_20170926_h18_m17_s03r023/'        
    cases[key]['label']  = 'GFS1d'                
    
    key  = '07-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_IKE_ATM2OCN_v3.0/rt_20170926_h18_m07_s13r299/'        
    cases[key]['label']  = 'GFS25d'                

if False:
    key  = '09-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a52_IKE_ATM2OCN_v3.0/rt_20170926_h18_m12_s08r851/'        
    cases[key]['label']  = 'GFS05d_OC_DA'    
    #key0   = key 
    #key0 = '01-atm:y-tid:y-wav:n'


###########################################
###########################################
wav_height_fname = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/HSOFS/V516/test3GFS05dOCDARegWind/ww3.WND.GFS05dOcDa.200809_hs.nc'
if False:
    key  = '02-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_IKE_ATM_WAV2OCN_v3.0/rt_20170926_h18_m03_s57r568/'        
    cases[key]['label']  = 'GFS05d_OC_Wav'         

if False:
    key  = '04-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_IKE_ATM_WAV2OCN_v3.0/rt_20170926_h17_m58_s55r771//'        
    cases[key]['label']  = 'GFS05d_Wav'                
                   
    key  = '06-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_IKE_ATM_WAV2OCN_v3.0/rt_20170926_h18_m19_s06r861/'        
    cases[key]['label']  = 'GFS1d_Wav'                
    
    key  = '08-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_IKE_ATM_WAV2OCN_v3.0/rt_20170926_h18_m09_s00r313/'        
    cases[key]['label']  = 'GFS25d_Wav'                
if True:
    key  = '10-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a53_IKE_ATM_WAV2OCN_v3.0/rt_20170926_h18_m14_s03r034/'        
    cases[key]['label']  = 'GFS05d_OC_DA_Wav'    
    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/EC2001V516WindZeta/V418/test2WindLevel/ww3.WindLevel20080910_hs.nc'
    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/Ike/WW3/StandAlone/GFS05d_OC_DA/ww3.WND.GFS05dOcDa.200809_hs.nc'
    cases[key]['wdir_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/EC2001V516WindZeta/V418/test2WindLevel/ww3.WindLevel20080910_dir.nc'

    print '\n\n\n\ IKE >>>> IKE  >>>>>>   hsig_file and wdir_file  is not correct !!!!!!!!!!!!!!!!!!! \n\n\n'
    key0 = key 
    key1 = key
    ###########
    key  = '11-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a60_IKE_fully_coupled/rt_test_dec13_transfer/'        
    cases[key]['label']  = 'Fully_coupled'    
    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/Ike/WW3/StandAlone/GFS05d_OC_DA/ww3.WND.GFS05dOcDa.200809_hs.nc'
    cases[key]['wdir_file'] = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/IKE/EC2001V516WindZeta/V418/test2WindLevel/ww3.WindLevel20080910_dir.nc'

    print '\n\n\n\ IKE >>>> IKE  >>>>>>   hsig_file and wdir_file  is not correct !!!!!!!!!!!!!!!!!!! \n\n\n'
    key0 = key 
    key1 = key


"""

    
