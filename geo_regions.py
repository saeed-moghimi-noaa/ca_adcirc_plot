#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geo regions for map plot
"""
__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2017, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"


from   pynmd.plotting.vars_param import *
from   pandas import read_csv
from   base_info import mesh_dir

def get_region_extent(region = 'hsofs_region'):
    if region == 'hsofs_region':
        defs['lim']['xmin']  = -99.0
        defs['lim']['xmax']  = -52.8
        defs['lim']['ymin']  =  5.0
        defs['lim']['ymax']  =  46.3
    ## READ WIND SWATH
    else: 
        df = read_csv(mesh_dir + region + '.txt',header=None)
        buff = 0.5
        defs['lim']['xmin']  = df.min()[0] - buff
        defs['lim']['xmax']  = df.max()[0] + buff
        defs['lim']['ymin']  = df.min()[1] - buff
        defs['lim']['ymax']  = df.max()[1] + buff

    return defs['lim']

'''  
    ##IKE
    elif region == 'caribbean':
        defs['lim']['xmin']  = -78. 
        defs['lim']['xmax']  = -74.
        defs['lim']['ymin']  =  20.
        defs['lim']['ymax']  =  24.
        
        defs['lim']['xmin']  = -82. 
        defs['lim']['xmax']  = -71.
        defs['lim']['ymin']  =  18.
        defs['lim']['ymax']  =  26.
    elif region == 'ike_region':
        defs['lim']['xmin']  = -98.5 
        defs['lim']['xmax']  = -84.5
        defs['lim']['ymin']  =  24.
        defs['lim']['ymax']  =  31.5
    elif region == 'caribbean_bigger':
        defs['lim']['xmin']  = -78.0
        defs['lim']['xmax']  = -58
        defs['lim']['ymin']  =  10.0
        defs['lim']['ymax']  =  28.
    elif region == 'ike_local':
        defs['lim']['xmin']  = -96 
        defs['lim']['xmax']  = -92
        defs['lim']['ymin']  =  28.5
        defs['lim']['ymax']  =  30.6
    elif region == 'ike_wave':
        defs['lim']['xmin']  = -95.63 
        defs['lim']['xmax']  = -88.0
        defs['lim']['ymin']  =  28.37
        defs['lim']['ymax']  =  30.50
    elif region == 'ike_hwm':
        defs['lim']['xmin']  = -96.15 
        defs['lim']['xmax']  = -88.5
        defs['lim']['ymin']  =  28.45
        defs['lim']['ymax']  =  30.7    
    elif region == 'ike_galv_bay':
        defs['lim']['xmin']  = -95.92 
        defs['lim']['xmax']  = -94.81
        defs['lim']['ymin']  =  29.37
        defs['lim']['ymax']  =  29.96
    elif region == 'ike_galv_nwm':
        defs['lim']['xmin']  = -95.4 
        defs['lim']['xmax']  = -94.2
        defs['lim']['ymin']  =  28.66
        defs['lim']['ymax']  =  30.4
    elif region == 'ike_wav_break':
        defs['lim']['xmin']  = -95 
        defs['lim']['xmax']  = -94.5
        defs['lim']['ymin']  =  28.7 + 0.6
        defs['lim']['ymax']  =  30.4 - 0.6       
    elif region == 'ike_f63_timeseries':
        defs['lim']['xmin']  = -94.2579  - 0.1
        defs['lim']['xmax']  = -94.2579  + 0.1
        defs['lim']['ymin']  =  29.88642 - 0.1
        defs['lim']['ymax']  =  29.88642 + 0.1      
    elif region == 'ike_f63_timeseries_det':
        defs['lim']['xmin']  = -94.2300
        defs['lim']['xmax']  = -94.1866
        defs['lim']['ymin']  =  29.82030
        defs['lim']['ymax']  =  29.84397+0.05    
    elif region == 'ike_cpl_paper':
        defs['lim']['xmin']  = -95.127481
        defs['lim']['xmax']  = -93.233053
        defs['lim']['ymin']  =  29.198490
        defs['lim']['ymax']  =  30.132224  
       
##IRMA
    elif region == 'carib_irma':
        defs['lim']['xmin']  = -84.0
        defs['lim']['xmax']  = -60.
        defs['lim']['ymin']  =  15.0
        defs['lim']['ymax']  =  29.
    elif region == 'burbuda':
        defs['lim']['xmin']  = -65.0
        defs['lim']['xmax']  = -60.
        defs['lim']['ymin']  =  15.0
        defs['lim']['ymax']  =  19.
    elif region == 'burbuda_zoom':
        defs['lim']['xmin']  = -63.8
        defs['lim']['xmax']  = -60.8
        defs['lim']['ymin']  =  16.8
        defs['lim']['ymax']  =  18.65
    elif region == 'puertorico':
        defs['lim']['xmin']  = -67.35
        defs['lim']['xmax']  = -66.531
        defs['lim']['ymin']  =  18.321
        defs['lim']['ymax']  =  18.674
    elif region == 'puertorico_shore':
        defs['lim']['xmin']  = -67.284
        defs['lim']['xmax']  = -66.350
        defs['lim']['ymin']  =  18.360
        defs['lim']['ymax']  =  18.890
    elif region == 'key_west':
        defs['lim']['xmin']  = -82.7
        defs['lim']['xmax']  = -74.5
        defs['lim']['ymin']  =  21.3
        defs['lim']['ymax']  =  27.2
    elif region == 'key_west_zoom':
        defs['lim']['xmin']  = -82.2
        defs['lim']['xmax']  = -79.4
        defs['lim']['ymin']  =  24.1
        defs['lim']['ymax']  =  26.1
    elif region == 'cuba_zoom':
        defs['lim']['xmin']  = -82.
        defs['lim']['xmax']  = -77.
        defs['lim']['ymin']  =  21.5
        defs['lim']['ymax']  =  23.5         
    elif region == 'key_west_timeseries':
        defs['lim']['xmin']  = -84.62
        defs['lim']['xmax']  = -79.2
        defs['lim']['ymin']  =  23.6
        defs['lim']['ymax']  =  30.0
    elif region == 'pr_timeseries':
        defs['lim']['xmin']  = -68
        defs['lim']['xmax']  = -64
        defs['lim']['ymin']  =  17.3
        defs['lim']['ymax']  =  19.2
    elif region == 'key_west_anim':
        defs['lim']['xmin']  = -85.5
        defs['lim']['xmax']  = -74.5
        defs['lim']['ymin']  =  21.0
        defs['lim']['ymax']  =  31.5        

    ## ISABEL
    elif region == 'isa_region':
        defs['lim']['xmin']  = -80.2 
        defs['lim']['xmax']  = -71.6
        defs['lim']['ymin']  =  31.9
        defs['lim']['ymax']  =  41.9
    
    elif region == 'isa_local':
        defs['lim']['xmin']  = -77.5
        defs['lim']['xmax']  = -74
        defs['lim']['ymin']  =  34.5
        defs['lim']['ymax']  =  40.0

        defs['lim']['xmin']  = -78.5
        defs['lim']['xmax']  = -74
        defs['lim']['ymin']  =  33.5
        defs['lim']['ymax']  =  39.5
    
    elif region == 'isa_hwm':
        defs['lim']['xmin']  = -76.01
        defs['lim']['xmax']  = -75.93
        defs['lim']['ymin']  =  36.74
        defs['lim']['ymax']  =  36.93    
    
    elif region == 'isa_landfall':
        defs['lim']['xmin']  = -77.8
        defs['lim']['xmax']  = -75.2
        defs['lim']['ymin']  =  34.2
        defs['lim']['ymax']  =  37.5
    elif region == 'isa_landfall_zoom':
        defs['lim']['xmin']  = -77.8
        defs['lim']['xmax']  = -75.2
        defs['lim']['ymin']  =  34.2
        defs['lim']['ymax']  =  36.0
    ## SANDY
    elif region == 'san_track':
        defs['lim']['xmin']  = -82.0
        defs['lim']['xmax']  = -67.0
        defs['lim']['ymin']  =  23.0
        defs['lim']['ymax']  =  43.6
    elif region == 'san_area':
        defs['lim']['xmin']  = -77.0
        defs['lim']['xmax']  = -70.0
        defs['lim']['ymin']  =  37.0
        defs['lim']['ymax']  =  42.0
        
    elif region == 'san_track':
        defs['lim']['xmin']  = -82.0
        defs['lim']['xmax']  = -67.0
        defs['lim']['ymin']  =  23.0
        defs['lim']['ymax']  =  43.6
    elif region == 'san_area':
        defs['lim']['xmin']  = -77.0
        defs['lim']['xmax']  = -70.0
        defs['lim']['ymin']  =  37.0
        defs['lim']['ymax']  =  42.0

    elif region == 'san_area2':
        defs['lim']['xmin']  = -75.9
        defs['lim']['xmax']  = -73.3
        defs['lim']['ymin']  =  38.5
        defs['lim']['ymax']  =  41.3
    elif region == 'san_newyork':
        defs['lim']['xmin']  = -74.5
        defs['lim']['xmax']  = -73.55
        defs['lim']['ymin']  =  40.35
        defs['lim']['ymax']  =  41.2
    elif region == 'san_delaware':
        defs['lim']['xmin']  = -75.87
        defs['lim']['xmax']  = -74.31
        defs['lim']['ymin']  =  38.26
        defs['lim']['ymax']  =  40.51    
    elif region == 'san_jamaica_bay':
        defs['lim']['xmin']  = -73.963520
        defs['lim']['xmax']  = -73.731455
        defs['lim']['ymin']  =  40.518074
        defs['lim']['ymax']  =  40.699618          
    elif region == 'irn_region':
        defs['lim']['xmin']  = -78.41
        defs['lim']['xmax']  = -73.48
        defs['lim']['ymin']  =  33.55
        defs['lim']['ymax']  =  41.31         
    elif region == 'irn_hwm':
        defs['lim']['xmin']  = -78.64
        defs['lim']['xmax']  = -69.54
        defs['lim']['ymin']  =  33.80
        defs['lim']['ymax']  =  41.82       
    
    
    
    ## ANDREW
    elif region == 'and_region':
        defs['lim']['xmin']  = -98.5 
        defs['lim']['xmax']  = -77.5
        defs['lim']['ymin']  =  23.
        defs['lim']['ymax']  =  32.
    elif region == 'and_fl_lu':
        defs['lim']['xmin']  = -98.5 
        defs['lim']['xmax']  = -76.5
        defs['lim']['ymin']  =  21.
        defs['lim']['ymax']  =  32.
    elif region == 'and_local_lu':
        defs['lim']['xmin']  = -95 
        defs['lim']['xmax']  = -86
        defs['lim']['ymin']  =  28.
        defs['lim']['ymax']  =  32
    elif region == 'and_local_fl':
        defs['lim']['xmin']  = -86 
        defs['lim']['xmax']  = -79.5
        defs['lim']['ymin']  =  24.
        defs['lim']['ymax']  =  34
    elif region == 'and_local_lu_landfall':
        defs['lim']['xmin']  = -92.4 
        defs['lim']['xmax']  = -87.5
        defs['lim']['ymin']  =  28.
        defs['lim']['ymax']  =  31.
    elif region == 'and_local_fl_landfall':
        defs['lim']['xmin']  = -80.0 
        defs['lim']['xmax']  = -80.5
        defs['lim']['ymin']  =  25.34
        defs['lim']['ymax']  =  25.8
    #MICHEAL
    elif region == 'fl_all':
        defs['lim']['xmin']  = -88
        defs['lim']['xmax']  = -79
        defs['lim']['ymin']  =  23.5
        defs['lim']['ymax']  =  32 
    elif region == 'mic_landfall':
        defs['lim']['xmin']  = -87.7
        defs['lim']['xmax']  = -81.31
        defs['lim']['ymin']  =  27.4
        defs['lim']['ymax']  =  31.01 
    elif region == 'mic_landfall_zoom':
        defs['lim']['xmin']  = -86.70
        defs['lim']['xmax']  = -83.00
        defs['lim']['ymin']  =  29.00
        defs['lim']['ymax']  =  31.01 

'''
