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

#base_dir_sm   = '/scratch2/COASTAL/coastal/noscrub/Yuji.Funakoshi/nsem-workflow/data/'
#track_fname   = '/scratch2/COASTAL/coastal/noscrub/Yuji.Funakoshi/nsem-workflow/parm/storms/harvey/bal092017.dat'
#hwm_fname     = 'xxx'
#base_dir_coops_piks = '/scratch2/COASTAL/coastal/noscrub/Yuji.Funakoshi/nsem-workflow/parm/storms/obs/'
#base_dir_obs        = '/scratch2/COASTAL/coastal/noscrub/Yuji.Funakoshi/nsem-workflow/parm/storms/obs/'

#nwm_channel_pnts    = 'xxx'
#nwm_channel_geom    = 'xxx'
#nwm_results_dir     = 'xxx'

#name = 'HARVEY'
#year = '2017'

#### INPUTS ####
base_dir_sm         = '/scratch2/NCEPDEV/stmp1/Andre.VanderWesthuysen/data/'
track_fname         = '/scratch2/COASTAL/coastal/noscrub/shared/Saeed.Moghimi/shared_with_Yuji/obs/best_tracks/florence_bal062018.dat'
hwm_fname           = '../obs/obs_all/hwm/florence2018.csv'
base_dir_obs        = '../obs/obs_all/'
nwm_channel_pnts    = None
nwm_channel_geom    = None
nwm_results_dir     = None
##
name = 'Florence'
year = '2018'


# map_plot_options
#### map_plot_options
plot_adc_fort       = True
plot_adc_maxele     = False
plot_nems_fields    = False
plot_forcing_files  = False
plot_nwm_files      = False
plot_transects      = False
plot_mesh           = False
vec                 = False
local_bias_cor      = True


#HWM proximity limit
prox_max = 0.0075 * 1  #grid size * n

if False:
    #Base run only tide
    key  = '00-atm:n-tid:y-wav:n'
    cases[key]['dir']    = '/scratch2/COASTAL/coastal/save/NAMED_STORMS/Florence_ADCIRC/hsofs120m/florence.tide_baserun.20210617/run/'
    cases[key]['label']  = 'Only tide'
    cases[key]['hsig_file'] = None
    cases[key]['wdir_file'] = None
    key0 = key

if False:
    #
    key  = '00-atm:y-tid:y-wav:n'

    cases[key]['dir']    = base_dir_sm + '/pahm-adcirc-florence_hsofs/run/'
    cases[key]['label']  = 'PAHM'
    cases[key]['hsig_file'] = None
    cases[key]['wdir_file'] = None
    key0 = key

if False:
    #
    key  = '00-atm:y-tid:y-wav:n'

    cases[key]['dir']    = '/scratch2/COASTAL/coastal/save/NAMED_STORMS/Florence_ADCIRC/run_20210812_florence_atmesh_120m_yujicopy_newnems/runs/unperturbed/'
    cases[key]['label']  = 'HWRF'
    cases[key]['hsig_file'] = None
    cases[key]['wdir_file'] = None
    key0 = key1 = key

if True:
    #2way wav-ocn
    key  = '00-atm:y-tid:y-wav:y'
    cases[key]['dir']    = base_dir_sm + '/pahm-adcirc-ww3-florence_hsofs.wav2ocn.success/run/'
    cases[key]['label']  = 'PAHM&WW3'
    cases[key]['hsig_file'] = None
    cases[key]['wdir_file'] = None
    key0 = key1 = key

    ###########

#out_dir  = '/scratch2/COASTAL/coastal/noscrub/Yuji.Funakoshi/ca_adcirc_plot/2017_harvey/'


out_dir  = '../pahm_plots/ca_adcirc_plot/Florence_pahm/'

#########################  Variable Limits Settings ####################################

tim_lim = {}
tim_lim['xmin'] = datetime.datetime(2018, 9, 10)  -  datetime.timedelta(2.0)
tim_lim['xmax'] = datetime.datetime(2018, 9, 18)  +  datetime.timedelta(2.0)


#### Deffinitions 
if True:
    # key1 - key0 >> key0 is tide
    defs['elev']['cmap']  =  maps.jetMinWi
    defs['elev']['label']  =  'Surge [m]'
    defs['elev']['vmin']  =  0
    defs['elev']['vmax']  =  5
    
else:
    # key1 - key0 >> key0 is run without wave
    defs['elev']['label']  =  'Wave set-up [m]'
    #defs['elev']['cmap']   =  maps.jetWoGn()
    defs['elev']['cmap']   =  plt.cm.jet
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
    defs['wind']['vmax']  =  40  
    defs['wind']['label'] = 'Wind Speed [m$^ \\mathrm{}$ s$^ \\mathrm{-1}$] '

defs['wind']['cmap']  =  maps.jetMinWi

################################################
varname = 'pmsl'
defs['pmsl']  ['fname']   = 'fort.73.nc'
#defs[varname]['label']   = 'Pressure [mH2O] '
defs[varname] ['label']  = 'pressure [Pa]'
defs[varname] ['var']     = 'pressure'
if True:
    defs['pmsl']['vmin']  = 9.2000
    defs['pmsl']['vmax']  = 10.5000
else:
    defs['pmsl']['vmin']  =  92000
    defs['pmsl']['vmax']  =  101000
    #defs[varname]['vmin']  =  val.min() // 1000 * 1000

defs['hs']['vmin']  =  0
defs['hs']['vmax']  =  8                    ### plot stuff
defs['rad' ]['fname'] = 'rads.64.nc'
defs['elev']['fname'] = 'fort.63.nc'
defs['wind']['fname'] = 'fort.74.nc'

##### SELECLT VARNAME TO PLOT ########################
#varnames = ['elev','rad','wind', 'hs','maxele']
#varnames = ['elev']
varnames = ['wind']
##### SELECLT GEO REGION TO PLOT ########################
#regions = ['isa_landfall_zoom'] #,'isa_landfall','isa_local','isa_region','hsofs_region']
#regions = ['ike_region','hsofs_region','ike_local','ike_galv_bay']
#regions = ['hsofs_region']

#regions = ['isa_landfall_zoom'] #,'isa_landfall','isa_local','isa_region','hsofs_region']
regions = ['hsofs_region']
#regions = ['isa_region']



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
"""


