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
storm_name = 'ISA'

# map_plot_options
plot_adc_fort       = False
plot_adc_maxele     = True
plot_nems_fields    = False
plot_forcing_files  = False
plot_nwm_files      = True
local_bias_cor      = True
plot_mesh           = False


#HWM proximity limit
prox_max = 0.01


base_dir_sm   = '/scratch4/COASTAL/coastal/noscrub/Saeed.Moghimi/01_stmp_ca/stmp7_isa/'
track_fname   = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/tracks/isabel_bal132003.dat'
hwm_fname     = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/hwm/events/hwm_isa.csv'
base_dir_coops_piks = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/coops/isabel/coops-isabel-data/ISA_data/sta_pickles_data/'
nwm_channel_pnts    = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/nwm_base_info/channel_points/NWM_v1.1_nc_tools_v1/spatialMetadataFiles/nwm_v1.1_geospatial_data_template_channel_point.nc'
nwm_channel_geom    = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/nwm_base_info/channel_geom/nwm_v1.1/nwm_fcst_points_comid_lat_lon-v1-1_ALL.csv'
nwm_results_dir     = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/wrfhydro/test_data/nwm.20180226/'

#base_dir_coops_piks = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/coops/isabel/coops-isabel-data/ISA_data/sta_pickles_all/'
#
#

mass_conserv_test = False
if mass_conserv_test:
    key  = '01-atm:n-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a21_ISA_TIDE_Mass_Conserv_v1.0/rt_20180130_h19_m56_s23r094_no_wet_dry/'        
    cases[key]['label']  = 'Tid_noWetDry'         
    key0   = key 
    
    
    key  = '02-atm:n-tid:y-wav:n'
    #cases[key]['dir']   = base_dir_sm + '/a21_ISA_TIDE_Mass_Conserv_v1.0/rt_20180129_h14_m11_s15r533/'  
    cases[key]['dir']    = base_dir_sm + '/a21_ISA_TIDE_Mass_Conserv_v1.0/rt_20180130_h14_m35_s02r091_wetdry/'  
    cases[key]['label']  = 'Tid_WetDry'         
    key1   = key 

    out_dir  = cases[key0]['dir'] +'../../01_post_'+cases[key0]['dir'].split('/')[-2]+'/'


##only wind forcing




Compare_coupled_vs_tide = True
Compare_ATMWAV_vs_ATM   = False

if Compare_coupled_vs_tide:
    #Base run only tide
    if False:
        key  = 'atm:n-tid:y-wav:n'
        cases[key]['dir']    = base_dir_sm + '/a21_ISA_TIDE_v1.0/rt_20170927_h19_m38_s41r196/'      
        cases[key]['label']  = 'Only tide' 
        #cases[key]['label']  = '' 
        key0   = key 
    
    key  = '01-atm:y-tid:y-wav:n'
    cases[key]['dir']    = base_dir_sm + '/a31_ISA_ATM2OCN_v1.0/rt_20190710_h21_m12_s16r376/'        
    cases[key]['label']  = 'ATM2OCN'         
    key0   = key 

    key  = '05-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/a51_ISA_ATM_WAV2OCN_test_merge_v1.0/rt_20190710_h21_m03_s43r363/'    
    wav_height_fname = '/scratch4/COASTAL/coastal/save/Ali.Abdolali/ISABEL/GoM/V516/test1NestLevCur/ww3.HWRFc.BND.LevCur.200309_hs.nc'              
    cases[key]['label']  = 'ATM&WAV2OCN'                
    key1   = key 

    if False:
        key  = '04-atm:y-tid:y-wav:y'
        cases[key]['dir']    = base_dir_sm + '/a51_ISA_ATM_WAV2OCN_v1.0/rt_20171010_h20_m47_s23r055/'    
        cases[key]['label']  = 'ATM&WAV2OCN-tide'                
        #key0   = key 
      
        key  = '02-atm:n-tid:y-wav:y'
        cases[key]['dir']    = base_dir_sm + '/a41_ISA_WAV2OCN_v1.0/rt_20170927_h17_m28_s20r224/'        
        cases[key]['label']  = 'WAV2OCN'         
        #key1   = key 
        
        key  = '05-atm:y-tid:y-wav:y'
        cases[key]['dir']    = base_dir_sm + '/a51_ISA_ATM_WAV2OCN_v1.0/rt_20171011_h16_m53_s02r857/'    
        cases[key]['label']  = 'ATM&WAV2OCN-tide l=0.03'                
        key1   = key 
        out_dir  = cases[key1]['dir'] +'01_post_no_ww3_force_limit_atm_diff/'
    
        key  = '03-atm:y-tid:y-wav:y'
        cases[key]['dir']    = base_dir_sm + '/a51_ISA_ATM_WAV2OCN_v1.0/rt_20170927_h17_m54_s04r019/' 
        cases[key]['label']  = 'ATM&WAV2OCN'                
        key0   = key 

    out_dir  = cases[key1]['dir'] +'../01_post/'




#########################  Variable Limits Settings ####################################

tim_lim = {}
tim_lim['xmin'] = datetime.datetime(2003, 9, 16 )
tim_lim['xmax'] = datetime.datetime(2003, 9, 20 )

#time of land fall
#tim_lim['xmin'] = datetime.datetime(2003, 9, 18,16 ) - datetime.timedelta(0.5)
#tim_lim['xmax'] = datetime.datetime(2003, 9, 18,16 ) + datetime.timedelta(0.5)

#time of land fall
#tim_lim['xmin'] = datetime.datetime(2003, 9, 18, 12 ) 
#tim_lim['xmax'] = datetime.datetime(2003, 9, 19, 12 ) 

tim_lim['xmin'] = datetime.datetime(2003, 9, 16 ) 
tim_lim['xmax'] = datetime.datetime(2003, 9, 21 ) 

if False:
    #maps anim
    tim_lim['xmin'] = datetime.datetime(2003, 9, 16 ) 
    tim_lim['xmax'] = datetime.datetime(2003, 9, 19,12 ) 
    
    
    tim_lim['xmin'] = datetime.datetime(2003, 9, 10 ) 
    tim_lim['xmax'] = datetime.datetime(2003, 9, 24 ) 


#for transport calc
#tim_lim['xmin'] = datetime.datetime(2003, 9, 10,18 ) 
#tim_lim['xmax'] = datetime.datetime(2003, 9, 14,17 ) 
#tim_lim['xmax'] = datetime.datetime(2003, 9, 24,17 ) 

#tim_lim['xmin'] = datetime.datetime(2003,  8,1) 
#tim_lim['xmax'] = datetime.datetime(2003, 12,1 ) 


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

#rad-stress
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

defs['hs']['vmin']  =  1
defs['hs']['vmax']  =  15                    ### plot stuff
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
#varnames = ['hs','wind']#,'elev'] #,'maxele']
#varnames = ['elev'] #,'maxele']
varnames = ['wind']#,'elev'] #,'maxele']
#varnames = ['hs']#,'elev'] #,'maxele']

##### SELECLT GEO REGION TO PLOT ########################
#regions = ['hsofs_region','ike_wave','caribbean']
#regions = ['ike_local','hsofs_region','ike_wave','ike_region',]#'caribbean']
#regions = ['ike_wave','ike_region',]
#regions = ['ike_region',]

#regions = ['hsofs_region']
#regions = ['caribbean']
#regions = ['ike_region']
#

#regions = ['isa_landfall_zoom','isa_landfall','isa_local']#,'isa_region','hsofs_region']
#regions = ['isa_region','hsofs_region']
#regions = ['isa_hwm','isa_region']
#regions = ['isa_local']
#regions = ['hsofs_region']
regions = ['isa_region']

#station_selected_list = None



#potomac river
#station_selected_list = ['8594900', '8635150', '8635750']




#isabel delaware#
station_selected_list =['8548989',  
                        '8538886',
                        '8539094',
                        '8540433',
                        '8537121',
                        '8551762',
                        '8555889',
                        '8536110',
                        '8557380',
                        '8534720',
#                        ]
#
#isabel chesepeak#
#station_selected_list =[ 
                         '8571892',#  
                         '8573364',# '8573927', 
                         '8574680', 
                         '8575512', 
                         '8577330', 
                         '8594900', 
                         #'8632837', 
                         '8635150', 
                         '8635750', 
                         '8636580', 
                         '8637624'
                         ]
# galveston
#station_selected_list = [   '8770777', '8770933', '8770971', '8771013', '8771328', '8771341', '8771450', '8771510', '8772440', '8772447']

