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
import os
import cmocean

os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join('BG/')

cases = defaultdict(dict)

#### INPUTS ####
home_dir      = '/lcrc/project/HSOFS_Ensemble/HSOFS/Subsetting_Paper/'
base_dir_sm   = home_dir + 'Florence/'
base_dir_obs  = home_dir + 'Florence/data/observations/'
hwm_fname     = home_dir + 'Florence/data/observations/Florence2018_HWM.csv'
mesh_dir      = home_dir + 'Florence/mesh/'
track_fname   = home_dir + 'Florence/data/Florence2018.22'
datum_fname   = home_dir + 'base_meshes/HSOFS_2016_msl2navd88.nc'
nwm_channel_pnts    = None
nwm_channel_geom    = None
nwm_results_dir     = None
##
name = 'FLORENCE'
year = '2018'
isotachs = ['34', '50', '64'] #[kt]

#HWM proximity limit
prox_max = 0.25 / 111  #[km] / [deg2km]

#Low frequency MSL offset estimate for hurricane period to add to maxele
reference_station = dict()
reference_station['fname'] = base_dir_obs + '8658163.csv'
reference_station['msl_offset'] = 0.658 #[m] 

#### Cases to compare

#original
key  = 'original'
cases[key]['dir']    = base_dir_sm + 'original/runs/unperturbed/'      
cases[key]['label']  = 'HSOFS 2016'
cases[key]['hsig_file'] = None
cases[key]['wdir_file'] = None
key0 = key 

#subset+merge meshes
for isotach in isotachs:
    key  = 'subset_' + isotach + 'kt'
    cases[key]['dir']    = base_dir_sm + key + '/runs/unperturbed/'      
    cases[key]['label']  = 'HSOFS 2016-' + isotach + 'kt'    
    cases[key]['hsig_file'] = None
    cases[key]['wdir_file'] = None

out_dir  = base_dir_sm  + '/output_figures/'

##### Variable Limits Settings 
tim_lim = {}
tim_lim['xmin'] = datetime.datetime(2018, 9, 11)  -  datetime.timedelta(2.0)
tim_lim['xmax'] = datetime.datetime(2018, 9, 16)  +  datetime.timedelta(2.0)

#### Definitions 
#max_ele
defs['elev']['cmap']  =  cmocean.cm.amp
defs['elev']['label'] =  'water level [m]'
defs['elev']['vmin']  =  0
defs['elev']['vmax']  =  5

#
defs['elev_diff']['cmap']  =  cmocean.cm.delta
defs['elev_diff']['label'] =  'water level difference [m]'
defs['elev_diff']['vmin']  =  -1
defs['elev_diff']['vmax']  =  +1

##### SELECT GEO REGION TO PLOT ########################
regions = ['HSOFS_2016_Florence2018_50kt_150m_subsetpoly']
#regions = ['hsofs_region']

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
