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
storm_name = 'IRE'

# map_plot_options
plot_adc_fort       = False
plot_adc_maxele     = True
plot_nems_fields    = False
plot_forcing_files  = False
plot_nwm_files      = False
plot_mesh           = False

#HWM proximity limit
prox_max = 0.01

base_dir_sm   = '/scratch4/COASTAL/coastal/noscrub/Saeed.Moghimi/stmp11_irene/'
hwm_fname     = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/hwm/events/hwm_irn.csv'
base_dir_coops_piks = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/ccxxxxxxxx/'
track_fname    = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/tracks/irene/al092011_best_track/bal092011.dat'


if False:
    #Base run only tide
    key  = 'atm:n-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a21_IRN_TIDE_v1.0/rt_20180518_h13_m06_s00r462/'      
    cases[key]['label']  = 'Only tide' 
    key0   = key 

#ATM2OCN
key  = '01-atm:y-tid:y-wav:n'
cases[key]['dir']    = base_dir_sm + '/a22_IRN_BEST_v1.0/rt_20180518_h14_m14_s12r656/'        
cases[key]['label']  = 'BestTrack'   
key0  = key 

if True:
    #ATM2OCN
    key  = '02-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a22_IRN_BEST_v1.0/rt_yuji_2011_irene/wind/'        
    cases[key]['label']  = 'BestTrack Yuji'   
    key1   = key 
    
if True:
    #ATM&WAV2OCN 
    wav_inp_dir = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/SANDY/WW3/'
    key  = '02-atm:y-tid:y-wav:y-try01'
    cases[key]['dir']    = base_dir_sm + '/a53_SAN_ATM_WAV2OCN_v1.0/XXXXXXXXXXXXXX/'        
    cases[key]['label']  = 'Try01+WAV'         
    cases[key]['hsig_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_hs.nc'
    cases[key]['wdir_file'] = wav_inp_dir + 'ww3.HWRF.3DVar.2012_dir.nc'
    #key1 = key

    cases[key]['hsig_file'] = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/Irene/WW3/ww3.HSOFS.T5.2011_hs.nc'
    cases[key]['wdir_file'] = '/scratch4/COASTAL/coastal/save/NAMED_STORMS/Irene/WW3/ww3.HSOFS.T5.2011_hs.nc'
    


out_dir  = cases[key1]['dir']+'/../../01_post_best2ocn/'
#######
defs['elev']['label']  =  'Elev [m]'
if True:
    defs['elev']['cmap']  =  maps.jetMinWi
    defs['elev']['label']  =  'Surge [m]'
    defs['elev']['vmin']  =  0
    defs['elev']['vmax']  =  4
else:
    defs['elev']['label']  =  'Wave set-up [m]'
    defs['elev']['cmap']   =  maps.jetWoGn()
    defs['elev']['vmin']  =  -0.5
    defs['elev']['vmax']  =   0.5

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
#regions = ['san_newyork','san_area2','hsofs_region','san_track','san_area']
#regions = ['hsofs_region','san_track','san_area2']
regions = ['irn_hwm','irn_region','san_delaware','san_area','san_track','san_area2']

#regions = ['san_track']

#regions = ['san_area']

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
   
    
    
