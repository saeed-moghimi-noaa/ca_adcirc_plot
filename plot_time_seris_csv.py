#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot multi axes ####



The data for statistics is seprated from timeseries plot.



"""
__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2017, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"


import os,sys
#sys.path.append('/home/moghimis/linux_working/00-working/04-test-adc_plot/')
#sys.path.append('/home/moghimis/linux_working/00-working/04-test-adc_plot/csdlpy')



import matplotlib
if os.name == 'nt':
    matplotlib.rc('font', family='Arial')
else:  # might need tweaking, must support black triangle for N arrow
    matplotlib.rc('font', family='DejaVu Sans')

import netCDF4
from   collections import defaultdict
# import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import datetime
import string
import glob
import time
import pandas as pd
import seaborn as sns
import itertools
import cftime

sns.set_style(style='dark')
#sns.set_style(style='ticks')

from   pynmd.plotting.vars_param import *
from   pynmd.plotting import plot_routines as pr
import pynmd.models.adcirc.post as adcp 
from   pynmd.tools.tide_analysis import tappy_filters
from   pynmd.plotting import colormaps as cmaps
import pynmd.plotting.plot_settings  as ps
import matplotlib.tri as Tri
from geo_regions import get_region_extent

try:
    os.system('rm base_info.pyc')
except:
    pass
if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
import base_info

tim_lim = base_info.tim_lim

plot_timeseries = True
pandas_plots    = True
plot_taylor     = True
plot_rtofs      = False
sub_tidal       = False
pandas_plots_time_series = True

print ('\n\n ##################################################################')
print ('   >>>>>>>>>>>>>>>>>   Plot timeseries                    ',plot_timeseries)
print ('   >>>>>>>>>>>>>>>>>   PANDAS_plots                       ',pandas_plots)
print ('   >>>>>>>>>>>>>>>>>   Plot  RTOFS                        ',plot_rtofs)
print ('   >>>>>>>>>>>>>>>>>   Plot  Sub-tidal                    ',sub_tidal)
print ('   >>>>>>>>>>>>>>>>>   Pre-selected stations to plot      ',base_info.station_selected_list)
print ('   >>>>>>>>>>>>>>>>>   pandas_plots_time_series           ',pandas_plots_time_series)
print (' ##################################################################\n\n\n\n\n')

# calculate statistics
elev_lim = 2.2  # m

############################
curr_time = time.strftime("%Y%m%d_h%H_m%M_s%S")
fs = 12
fs2 = 10       

# plt.rc('font', family='serif')
plt.rcParams.update(\
        {'axes.labelsize': fs,
         #'text.fontsize': fs,
         'xtick.labelsize': fs,
         'ytick.labelsize': fs,
         'axes.titlesize': fs,
         'legend.fontsize': fs / 1.5,
         })

def read_csv(obs_dir, name, year, label):
    """
    examples
    print('  > write csv files')
    write_csv(base_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
    write_csv(base_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
    write_csv(base_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )
    write_csv(base_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
    
    """
    outt    = os.path.join(obs_dir, name+year,label)
    outd    = os.path.join(outt,'data')  
    if not os.path.exists(outd):
       sys.exit('ERROR',outd )

    table = pd.read_csv(os.path.join(outt,'table.csv')).set_index('station_name')
    table['station_code'] = table['station_code'].astype('str')
    stations = table['station_code']

    data     = []
    metadata = []
    for ista in range(len(stations)):
        sta   = stations [ista]
        fname8 = os.path.join(outd,sta)+'.csv'
        df = pd.read_csv(fname8,parse_dates = ['date_time']).set_index('date_time')
        
        fmeta = os.path.join(outd,sta) + '_metadata.csv'
        meta  = pd.read_csv(fmeta, header=0, names = ['names','info']).set_index('names')
        
        meta_dict = meta.to_dict()['info']
        meta_dict['lon'] = float(meta_dict['lon'])
        meta_dict['lat'] = float(meta_dict['lat'])        
        df._metadata = meta_dict
        data.append(df)
    
    return table,data

def get_station_ssh(fort61,lim):
    """
        Read model ssh
    """
    nc0      = netCDF4.Dataset(fort61)
    ncv0     = nc0.variables 
    sta_lon  = ncv0['x'][:]
    sta_lat  = ncv0['y'][:]
    sta_nam  = ncv0['station_name'][:].squeeze()
    sta_zeta = ncv0['zeta']        [:].squeeze()
    sta_date = netCDF4.num2date(ncv0['time'][:], ncv0['time'].units)

    stationIDs = []
    mod    = []
    ind = np.arange(len(sta_lat))

    [ind_bbox] = np.where((sta_lon > lim['xmin']) & 
                    (sta_lon < lim['xmax']) & 
                    (sta_lat > lim['ymin']) & 
                    (sta_lat < lim['ymax']))
    
    if False:
        indi = ind
    else:
        indi = ind_bbox   
    
    for ista in indi:
        stationID = sta_nam[ista].tostring().decode().rstrip()
        stationIDs.append(stationID)
        mod_tmp = pd.DataFrame(data = np.c_[sta_date, sta_zeta[:,ista]], columns=['date_time',  'ssh']).set_index('date_time')
        mod_tmp._metadata = stationID
        mod.append(mod_tmp)

    stationIDs = np.array(stationIDs)
    
    data = np.c_[indi, stationIDs]
    mod_table = pd.DataFrame(data = data, columns=['ind',  'station_code'])
   
    return mod,mod_table

###################################################################
def return_index(pattern, plist):
    for i in range(len(plist)):
        if pattern in plist[i]:
            return i
            break
    return None



#############################################################
def some_thing():
    ############# Sea Surface height analysis ########################
    # For simplicity we will use only the stations that have both wind speed and sea surface height and reject those that have only one or the other.
    common  = set(ssh_table['station_code']).intersection(mod_table  ['station_code'].values)
    
    ssh_obs, ssh_mod = [], []
    for station in common:
        ssh_obs.extend([obs for obs in ssh if obs._metadata['station_code'] == station])
        ssh_mod.extend([obm for obm in mod if obm._metadata                 == station])
    
    
    index = pd.date_range(
        start = start_dt.replace(tzinfo=None),
        end   = end_dt.replace  (tzinfo=None),
        freq=freq
    )
    
    #############################################################
    #organize and re-index both observations
    # Re-index and rename series.
    ssh_observations = []
    for series in ssh_obs:
        _metadata = series._metadata
        obs = series.reindex(index=index, limit=1, method='nearest')
        obs._metadata = _metadata
        obs.dropna(inplace=True)
        obs.name = _metadata['station_name']
        ssh_observations.append(obs)
    
    ##############################################################
    #model
    ssh_from_model = []
    for series in ssh_mod:
        _metadata = series._metadata
        obs = series.reindex(index=index, limit=1, method='nearest')
        obs._metadata = _metadata
        obs.name = _metadata
        obs['ssh'][np.abs(obs['ssh']) > 10] = np.nan
        obs.dropna(inplace=True)
        ssh_from_model.append(obs)
    
    for ssh1, model1 in zip(ssh_observations, ssh_from_model):
        fname = ssh1._metadata['station_code']
        location = ssh1._metadata['lat'], ssh1._metadata['lon']
        p = make_plot(ssh1, model1, label='SSH [m]',remove_mean_diff=remove_mean_diff, bbox_bias = bbox_bias)
        #p = make_plot(ssh1, ssh1)    
        marker = make_marker(p, location=location, fname=fname)
        marker.add_to(marker_cluster_coops)
        #del ssh_observations, ssh_from_model


###################################################
def get_rtofs_elev(stationID):
    """
    inp: filename, station_name
    out: date and elev
    """
    fort61 = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/data/ARTOFS-ike/rtofs_fort61.nc'
    fort61_sta_ids = get_station_ids(fort61)
    nc = netCDF4.Dataset(fort61)
    ncv = nc.variables 
    dates = netCDF4.num2date(ncv['time_rtofs'][:], ncv['time_rtofs'].units)
    ind2 = return_index (stationID, fort61_sta_ids)
    elev = ncv['zeta_rtofs'][:, ind2]
    nc.close()
    return  dates, elev

def datetime64todatetime(dt):
    tmp = []
    for it in range(len(dt)):
        d = dt[it]
        if isinstance(d, cftime.DatetimeGregorian):
            tmp.append(datetime.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second))
        else:
            tmp.append(pd.Timestamp(dt[it]).to_pydatetime())
    return np.array(tmp)

def do_tappy_tide_analysis(dates, elev):
    # filtered_dates, result = tappy_filters('transform', dates, elev)

      obs = pd.DataFrame(elev, columns=['elev'])
      se = pd.Series(dates)
      obs = obs.set_index (se)   
      obs.dropna()
      obsh = obs.resample('H').mean()


def do_tappy_tide_analysis(dates, val):
    """
    
    
    """
    obs = pd.DataFrame(val, columns=['val'])
    se = pd.Series(dates)
    obs = obs.set_index (se)   
    obs.dropna()
    obsh = obs.resample('H').mean()
   
    dates = datetime64todatetime(obsh.index)
    val = obsh.val
    # ## 
    from tappy.tappy import tappy   
    # ## Saeed tries to understand!  from here
    data_filename = 'test'
    def_filename = None
    config = None
    quiet = False
    debug = False
    outputts = False
    outputxml = ''
    ephemeris = False
    rayleigh = 0.9
    print_vau_table = False
    missing_data = 'ignore'
    # missing_data    = 'fill'
    linear_trend = False
    remove_extreme = False
    zero_ts = None
    filter = None
    pad_filters = None
    include_inferred = True
    xmlname = 'test'
    xmlcountry = 'US'
    xmllatitude = 100
    xmllongitude = 100
    xmltimezone = '0000'
    xmlcomments = 'No comment'
    xmlunits = 'm or ms-1'
    xmldecimalplaces = None
    
    ############## model
    x = tappy(
        outputts=outputts,
        outputxml='model.xml',
        quiet=quiet,
        debug=debug,
        ephemeris=ephemeris,
        rayleigh=rayleigh,
        print_vau_table=print_vau_table,
        missing_data=missing_data,
        linear_trend=linear_trend,
        remove_extreme=remove_extreme,
        zero_ts=zero_ts,
        filter=filter,
        pad_filters=pad_filters,
        include_inferred=include_inferred,
        )
    
    x.dates = dates
    x.elevation = val
    package = x.astronomic(x.dates)
    (x.zeta, x.nu, x.nup, x.nupp, x.kap_p, x.ii, x.R, x.Q, x.T, x.jd, x.s, x.h, x.N, x.p, x.p1) = package
    ray = 1.0
    (x.speed_dict, x.key_list) = x.which_constituents(len(x.dates), package, rayleigh_comp=ray)
    
    # x.constituents()
    # x.print_con()
    # x.print_con_file(filedat = out_file, lon = lon, lat = lat)
    x.dates_filled, x.elevation_filled = x.missing(x.missing_data, x.dates, x.elevation)
    x.write_file(x.dates_filled,
        x.elevation_filled,
        fname='outts_filled.dat')

    x_dates_filter, x_eleva_filter = x.filters('transform', x.dates, x.elevation)
    # x.filter='usgs'
    # x.filter='cd'
    # x.filter='boxcar'
    # x.filter='doodson'
    # x.filter='transform'        
    # if x.filter:
    #        for item in x.filter.split(','):
    #            if item in ['mstha', 'wavelet', 'cd', 'boxcar', 'usgs', 'doodson', 'lecolazet1', 'kalman', 'transform']:# 'lecolazet', 'godin', 'sfa']:
    #                filtered_dates, result = x.filters(item, x.dates, x.elevation)
    #                x.write_file(filtered_dates, result, fname='outts_filtered_%s.dat' % (item,))
    #                x_dates_filter= filtered_dates
    #                x_eleva_filter= result              
    #        (x.speed_dict, x.key_list) = x.which_constituents(len(x.dates),package,rayleigh_comp = ray)
    
    units = "seconds since 1970-01-01 00:00:00 UTC"
    yd_time = x.dates 
    yd_sec = netCDF4.date2num(yd_time        , units)
    ft_sec = netCDF4.date2num(x_dates_filter , units)
    ft_lev_new = np.interp(yd_sec, ft_sec, x_eleva_filter)
    elev_filter = ft_lev_new
    
    return  yd_time , ft_lev_new


#sys.path.append('/scratch2/COASTAL/coastal/save/Saeed.Moghimi/opt/pycodes/csdlpy')
#import adcirc
from atcf import readTrack

def plot_track(ax,track,date=None,color = 'r'):
    if date is not None:
        dates = np.array(track['dates'])
        #ind = np.array(np.where((dates==date))).squeeze().item()
        ind = find_nearest_time(dates,date)
        
        ax.plot(track['lon'][ind],track['lat'][ind],'ro',alpha=1,ms=8)
    
    ax.plot(track['lon'],track['lat'],lw=3,color=color,ls='dashed',alpha=1)

def read_track(fname=None):
    if fname is None:
        fname = '/scratch2/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/tracks/ike_bal092008.dat'
    
    track = readTrack(fname)
    keys = ['dates', 'lon', 'vmax', 'lat']
    for key in keys:
        tmp   = pd.DataFrame(track[key],columns=[key])

        #dfh   = df
        if 'trc' not in locals():
            trc = tmp
        else:
            trc  = pd.concat([trc,tmp],axis=1)    
    
    trc = trc.drop_duplicates(subset='dates',keep='first')
    trc = trc.set_index (trc.dates)
    trc = trc.resample('H').interpolate()
    trc.drop('dates',axis=1,inplace=True)
    
    dates = datetime64todatetime(trc.index)
    
    return dict(dates=dates,lon=trc.lon.values, lat=trc.lat.values)

def plot_map(sta_tab=[],comm_tab=[],prefix='',iplot=0):
        print ('Plot map for recording points ...')
        #construct HSOFS Tri mask
        fname  = base_info.cases[base_info.key1]['dir'] + '/maxele.63.nc'
        nc0    = netCDF4.Dataset(fname)
        ncv0   = nc0.variables
        depth  = ncv0['depth'][:]
        depth [depth.mask]  = -10.0
        lon0   = ncv0['x'][:]
        lat0   = ncv0['y'][:]
        elems  = ncv0['element'][:,:]-1  # Move to 0-indexing by subtracting 1
        tri = Tri.Triangulation(lon0,lat0, triangles=elems)

        nc0.close()
        
        fig, ax = adcp.make_map(res = None)
        fig.set_size_inches(7,7)
        lim = get_region_extent(region = base_info.regions[0])

        extent = [lim['xmin'],lim['xmax'],lim['ymin'],lim['ymax']+0]        
        ax.set_extent(extent)

        try:
            track = read_track(fname = base_info.track_fname)
            plot_track(ax,track,date=None)
        except:
            pass        
        
        cond1 = ax.tricontour(tri,depth+0.1   ,levels=[0.0]  ,colors='k',lw=0.02, alpha=0.8)

        #plot mesh
        if False:
            ax.triplot(tri, 'w-', lw=0.1, alpha=0.4)
        
        #plot all stations
        for il in range(len(comm_tab)):
            ax.scatter(comm_tab.iloc[il]['lon'],comm_tab.iloc[il]['lat'],s=15,c='y',zorder=3,edgecolors='None')
        
        #plot these stations
        for il in range(len(sta_tab)):
            ax.scatter(sta_tab.iloc[il]['lon'],sta_tab.iloc[il]['lat'],s=50,c='r',zorder=3,marker=ps.marker[il],label=sta_tab.iloc[il]['station_code'])
            #ax.plot(ncv['x'][ind2],ncv['y'][ind2],color='r')
         
        plt.legend(loc=4,ncol=4,frameon=True,numpoints=1,scatterpoints=1,prop={'size':8},framealpha=0.5)
        #ax.set_title(header)   
        filename = out_dir + '/' +str(100+iplot)+'_map_' +prefix + '_' + '.png'
        plt.savefig(filename,dpi=dpi//1.4)
        
        plt.close('all')

#### END of Funcs


#read Obs data
print (' read from OBS CSV files')
#try:
print ('   >  ssh from CSV files')
ssh_table,ssh = read_csv (base_info.base_dir_obs, base_info.name, base_info.year, label='coops_ssh' )

"""
except:
    print ('     >  No ssh CSV files')            

try:
    print ('   >  wind from CSV files')
    wnd_obs_table,wnd_obs  = read_csv (base_info.base_dir_obs, name, year, label='coops_wind')
except:
    print ('     >  No wind CSV files')  

try:
    print ('   >  NDBC wind from CSV files')
    wnd_ocn_table,wnd_ocn  = read_csv (base_info.base_dir_obs, name, year, label='ndbc_wind' )
except:
    print ('     >  No ndbc wind CSV files')  

try:
    print ('   >  NDBC wave from CSV files')
    wav_ocn_table, wav_ocn = read_csv (base_info.base_dir_obs, name, year, label='ndbc_wave' )
except:
    print ('     > no NDBC wave CSV files')  
"""

for x in base_info.cases[base_info.key0]['dir'].split('/'):
    if 'rt_' in x:  
        prefix = x
    else:
        prefix = ''.join(base_info.cases[base_info.key0]['dir'].split('/')[-3:])


prefix = 'time_series_' + prefix 

        
t1 = tim_lim['xmin'].isoformat()[2:13] + '_' + tim_lim['xmax'].isoformat()[2:13] + '_' + curr_time

out_dir = base_info.out_dir + prefix + t1 + '/'
# out dir and scr back up
scr_dir = out_dir + '/scr/'
os.system('mkdir -p ' + scr_dir)
args = sys.argv
scr_name = args[0]
os.system('cp -fr  ' + scr_name + '    ' + scr_dir)
os.system('cp -fr        *.py     '      + scr_dir)
print (out_dir)

#####################################################################
# graphic settings
ftypes = ['png', 'pdf']
#
left1 = 0.1  # the left side of the subplots of the figure
right1 = 0.9
bottom1 = 0.3  # the bottom of the subplots of the figure   (ntr==16   bottom=0.05)
top1 = 0.9
wspace1 = 0.1
hspace1 = 0.15
#
dpi = 300

#for map plot
lon,lat,tri  = adcp.ReadTri(base_info.cases[base_info.key1]['dir'])
####################################################3###############
#
defs['elev']['cmap'] = cmaps.jetWoGn()
#
defs['elevdif']['vmin'] = -0.05
defs['elevdif']['vmax'] = 0.05
#
defs['hs']['vmax'] = 2
defs['hs']['vmin'] = 0.2
#
params = ['elev']
defs['elev']['label']  =  'Elev. [m]'

#list of COOPS data files
#fpiks = np.array(glob.glob(base_info.base_dir_coops_piks + '/*.pik'))


#sta_coops_list = []
#for fpik in fpiks:
#    stationID_new = fpik.split('/')[-1][0:7]
#    sta_coops_list.append(stationID_new)

#get stations in bounding box

defs['lim'] = get_region_extent(region = base_info.regions[0] )
#ind_sta_all, ncv0, stationIDs_model = get_station_list(fort61, defs['lim'])

#sta_old_list = []
#for fpik in fpiks:
#    stationID_new = fpik.split('/')[-1][0:7]
#    sta_old_list.append(stationID_new)


if base_info.station_selected_list is not None:
    stationIDs = np.array(base_info.station_selected_list)
else:
    #stationIDs = np.array(ssh_table.station_code)
    ##############3
    #This is just to generate the common list of stations
    fort61 = base_info.cases[np.sort(list(base_info.cases.keys()))[0]]['dir'] + '/fort.61.nc'
    mod , mod_table = get_station_ssh(fort61=fort61,lim=defs['lim'] )
    tmp_ssh_table = ssh_table.sort_values('lat')
    common  = set(mod_table['station_code']).intersection(tmp_ssh_table  ['station_code'].values)
    stationIDs  = np.sort(list(common))

#Get matched stations lists
#sta_list_all = return_matched_items(stationIDs_model,stationIDs)
sta_list_all = stationIDs


all_stations = {}


if plot_timeseries:
    icol = 2
    irow_max = 2
    
    
    nplot = len(sta_list_all)// (irow_max * icol)
    iloc = 0
    for iplot in range(nplot):
        if len(sta_list_all) < (irow_max * icol):
            print (' ... All list')
            sta_list = sta_list_all
            # match the numebr to clumns
            sta_list = sta_list[:n]
        else:
            print (' ... partial list')

            sta_list = sta_list_all[iloc:iloc+icol*irow_max]         
            iloc     = iloc + icol * irow_max


        n = len(sta_list) // icol * icol
        irow = len(params) * len(sta_list) // icol    

    
        fwidth = icol * 10 
        fheight = irow * 4
        #
        fig, axgrid = plt.subplots(nrows=irow, ncols=icol, sharex=True, sharey=False,
                figsize=(fwidth, fheight), facecolor='w', edgecolor='k', dpi=dpi)
        #
        plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
                    wspace=wspace1, hspace=hspace1)
        #
        axgrid = np.array(axgrid)
        axgrid = axgrid.reshape(icol * irow)
        #
        nn = 0
        #sta_list = []
        for sta in sta_list:
            for param in params:
                ax = axgrid[nn]
                ic = 0
        
                defs['elev']['vmin'] = -0.5
                defs['elev']['vmax'] =  0.5
                cl = itertools.cycle(sns.color_palette(palette='dark', n_colors=2 * len(base_info.cases.keys())))
                # cl = itertools.cycle(sns.color_palette(palette = 'mute', n_colors = 2 * len(base_info.cases.keys()) ) )
                
                keys = np.sort(list(base_info.cases.keys()))
                for key in keys:
                        # try:
                        print (key)
                        fort61 = base_info.cases[key]['dir'] + '/fort.61.nc'
                        mod , mod_table = get_station_ssh(fort61,defs['lim'])
     
                        
                        ############# Sea Surface height analysis ########################
                        # For simplicity we will use only the stations that have both wind speed and sea surface height and reject those that have only one or the other.
                        #tmp_ssh_table = ssh_table.sort_values('lat')

                        common  = set(ssh_table['station_code']).intersection(mod_table  ['station_code'].values)
                        #common  = set(mod_table['station_code']).intersection(tmp_ssh_table  ['station_code'].values)
                        common  = np.sort(list(common))
                        #ssh_obs, ssh_mod , comm_tab = [], [], []
                        #for station in common:
                        #    ssh_obs.extend([obs for obs in ssh   if obs._metadata['station_code'] == station])
                        #    ssh_mod.extend([obm for obm in mod   if obm._metadata                 == station])
                       

                        comm_tab = ssh_table[ssh_table['station_code'].isin(common)]   


                        #station_names, stationIDs = [],[]
                        #for obs in ssh_obs:
                        #    station_names.append(obs._metadata['station_name'])
                        #    stationIDs.append(obs._metadata['station_code'])

                        stationID = sta#stationIDs[ista]
                        #sta_list.append(stationID)
                        print (stationID)
                        indo = return_index (stationID, ssh_table['station_code'].values)
                        indm = return_index (stationID, mod_table['station_code'].values)
                        header = 'Station: ' + stationID + ' at ' + ssh_table.index[indo]
                        prefix =  header.replace(':','_').replace(' ','_').replace(',','_').replace(',','_')
                    
                        print (header)
     
                        defs['elev']['vmin'] = 1.2 * max(ssh[indo].max().values,mod[indm].max().values ).astype(float)
                        defs['elev']['vmax'] = 1.1 * min(ssh[indo].min().values,mod[indm].min().values ).astype(float)
                        
                        
                        # model
                        dates1 = datetime64todatetime(mod[indm].index) 
                        val1   = mod[indm].values 
                        
                        #model data bias
                        bias = np.mean(ssh[indo].values)- np.mean(val1)
                                                                      
                        if base_info.local_bias_cor:
                            val1 = val1 + bias
                        
                        
                        data = dict(xx  =  dates1       ,
                                    val =  val1         ,
                                    #
                                    var = defs[param],
                                    lim = tim_lim,
                                    )
                        #
                        args = dict(
                                    title=header,
                                    label=base_info.cases[key]['label'],
                                    color=ps.cl[ic],
                                    # color      = next(cl),                          
                                    plot_leg=True,
                                    #panel_num = nn,
                                    )
                        #
                        if sub_tidal:
                            print ('sub_tidal')
                            dates1, val1 = do_tappy_tide_analysis(dates, val)
                            data['xx'] = dates1
                            data['val'] = val1
                            pr.TimeSeriesPlot(ax, data, args)
                        else: 
                            pr.TimeSeriesPlot(ax, data, args)
        
                        # ax.plot(dates,val)
                        ic += 1
                        # except:
                        # print ('        >      unsucessful key >' , key)
                        # #sys.exit()
                        # pass  
                  
                # plot OBS station data
                dates = datetime64todatetime(ssh[indo].index) 
                val   = ssh[indo].values          
                
                data = dict(
                      xx  = dates[::5]       ,
                      val = val  [::5]           ,
                      #
                      # xx   = stations['tides']['dates']       ,
                      # val  = stations['tides']['values']            ,
                      #
                      # xx   = stations['wl_obs']['dates']       ,
                      # val  = stations['wl_obs']['values']            ,
                    
                      var=defs[param],
                      lim=tim_lim,
                      )
                #
                args = dict(
                      title=header,
                      label='Data',
                      color='k',
                      plot_leg=True,
                      obs=True,
                      panel_num = nn,
                      ms=4,
                      alpha = 0.5
                      )
                #
                
                no_data = False
                if not no_data:
                    # pr.TimeSeriesPlot(ax,data,args)
                    if sub_tidal:
                        dates1, val1 = do_tappy_tide_analysis(dates, val)
                        data['xx'] = dates1
                        data['val'] = val1
                        pr.TimeSeriesPlot(ax, data, args)
                    else: 
                        pr.TimeSeriesPlot(ax, data, args)
        
                    if plot_rtofs:
                        dates, val = get_rtofs_elev(stationID)    
                        dates = dates - datetime.timedelta(1)  # dates changed without any reason  
                        key = 'artofs'
                        stations[key] = {}
                        stations[key]['dates'] = dates
                        stations[key]['values'] = val
          
                        data = dict(xx=dates           ,
                                    val=val            ,
                                    #
                                    var=defs[param],
                                    lim=tim_lim,
                                    )
                        #
                        args = dict(
                              label='ARTOFS',
                              color=ps.cl[ic + 1],
                              plot_leg=True,
                              )
                        
                        # pr.TimeSeriesPlot(ax,data,args)
                        
                        if sub_tidal:
                            dates1, val1 = do_tappy_tide_analysis(dates, val)
                            data['xx'] = dates1 
                            data['val'] = val1
                            pr.TimeSeriesPlot(ax, data, args)
                        else: 
                            pr.TimeSeriesPlot(ax, data, args)       
               
                nn += 1         
                
                #if not no_data:
                #    all_stations[stationID] = stations

        plt.savefig(out_dir + '/' +str(100+iplot)+ '_tim_' +prefix + '_' + '.png', dpi=dpi) 
        #plt.savefig(out_dir + '/' +str(100+iplot)+ '_tim_' +prefix + '_' + '.pdf') 

        plt.close('all')

        ##################################################################################
        sta_tab = ssh_table[ssh_table['station_code'].isin(sta_list)]   

        plot_map(sta_tab=sta_tab,comm_tab=comm_tab,prefix=prefix,iplot=iplot)
    
# if plot_rtofs:
#    base_info.cases['artofs'] = {}




if False:
    ###########################################################################################
    ###########################################################################################
    def model_on_data(data_dates, model_dates, model_val):
        print ('  >>>>>>>>>>>>>>>   ')
        units     = 'seconds since 2012-04-01 00:00:00'
        data_sec  = netCDF4.date2num(data_dates , units)
        model_sec = netCDF4.date2num(model_dates, units)
        return np.interp(data_sec, model_sec, model_val)
        
    # read and construct timeseries
    all_stations = {}
    nn = 0
    for ista in range(len(sta_list_all)):
        stationID = sta_list_all[ista]
        print (stationID)
        ind1 = return_index (stationID, fpiks)
        if ind1 is not None:
            sta_pik = fpiks   [ind1]
        else:
            print (' Remove this station from the list: no data is avaiable for the time of storm > ' , stationID)
            sys.exit(' ERROR !')
        f2 = file(sta_pik , 'r')
        stations = pickle.load(f2)    
        f2.close()  
        
        header = 'Station: ' + stationID + ' at ' + stations['info']['name'] + ', ' + stations['info']['state']
        print   (stations['info']['name'], stations['info']['lon'], stations['info']['lat'], ' ! ', stationID)
        
        
        for param in params:
            keys = np.sort(base_info.cases.keys())
            for key in keys:
                print (key)
                fort61 = base_info.cases[key]['dir'] + '/fort.61.nc'
                fort61_sta_ids = get_station_ids(fort61)
                nc = netCDF4.Dataset(fort61)
                ncv = nc.variables 
                dates = netCDF4.num2date(ncv['time'][:], ncv['time'].units)
                ind2 = return_index (stationID, fort61_sta_ids)
                val = ncv[defs['elev']['var']][:, ind2]
                nc.close()
                
                #
                stations['cwlev']['dates' ] = np.array(stations['cwlev']['dates' ]).squeeze()
                stations['cwlev']['values'] = np.array(stations['cwlev']['values']).squeeze()            
                
                stations[key] = {}
                stations[key]['dates'] = dates
                stations[key]['values'] = val
                #
    
            if len(stations['cwlev']['values']) > 1:
                all_stations[stationID] = stations
    
               
    print ('   >>>  Map for Taylor diagram')
    plot_map(sta_list_all=all_stations.keys(),sta_list=[],prefix='for_taylor',iplot=0)
    
    
    
    from pynmd.tools.compute_statistics import statatistics
    
    taylor_data = {}
    for sta in all_stations.keys():
        # find stations with exterme levels
        max_elev = 0
        keys = np.sort(base_info.cases.keys())
        for key in keys:
            max_elev = max (max_elev, max(all_stations[sta]['cwlev']['values']), all_stations[sta][key]['values'].max())
            
            base_info.cases[key]['model_on_data'] = []
            base_info.cases[key]['data_on_data'] = []
            base_info.cases[key]['data_on_data_stations'] = []
    
        # interpolate model on data points
        if max_elev > elev_lim:
            keys = np.sort(base_info.cases.keys())
            for key in keys:   
                ind = np.where ((np.array(all_stations[sta]['cwlev']['dates']) >= tim_lim['xmin']) & 
                                (np.array(all_stations[sta]['cwlev']['dates']) <= tim_lim['xmax']))
                #
                ind = np.array(ind).squeeze()
                all_stations[sta][key]['val_on_data'] = model_on_data(
                    data_dates   = all_stations[sta]['cwlev']['dates'][ind],
                    model_dates  = all_stations[sta][key]['dates'],
                    model_val    = all_stations[sta][key]['values'])
                
                base_info.cases[key]['model_on_data'].append(all_stations[sta][key]['val_on_data'])
                base_info.cases[key]['data_on_data' ].append(all_stations[sta]     ['cwlev']['values' ][ind])
                base_info.cases[key]['data_on_data_stations'].append(sta)
            
    # constract model and data for statistics calculation    
    keys = np.sort(base_info.cases.keys())
    for key in keys:
        model = np.array(base_info.cases[key]['model_on_data']).squeeze()
        data = np.array(base_info.cases[key]['data_on_data' ]).squeeze()
    
        base_info.cases[key]['stats'] = statatistics (data, model)
        taylor_data.update({base_info.cases[key]['label']:[model.std(ddof=1), np.corrcoef(data, model)[0, 1]]})
    
    
    if plot_taylor:
        #####
        print ('Plot Taylor Dig.')
        from pynmd.plotting import taylor
        markersize = 8
        fig = plt.figure(6, figsize=(9, 9))
        fig.clf()
    
        refstd = data.std(ddof=1)
    
        # Taylor diagram
        dia = taylor.TaylorDiagram(refstd, fig=fig, rect=111, label="Reference")
         
        colors = plt.matplotlib.cm.jet(np.linspace(0, 1, len(taylor_data.keys())))
         
        # Add samples to Taylor diagram
        for imodel in range(len(taylor_data.keys())):
            key = taylor_data.keys()[imodel]
            stddev = taylor_data[key][0]
            corrcoef = taylor_data[key][1]
            marker = ps.marker[imodel]
            dia.add_sample(stddev, corrcoef, ref=False, marker=marker, ls='', c=ps.colors[imodel],
                       markersize=markersize, label=key)
             
        # add refrence point for data     
        dia.add_sample(refstd, 1.0 , ref=True, marker='*', ls='', c='k',
                       markersize=markersize * 1.5, label='Ref.')
         
        # Add RMS contours, and label them
        contours = dia.add_contours(levels=8, data_std=refstd, colors='0.5')
        plt.clabel(contours, inline=1, fmt='%.3g', fontsize=10)
         
        dia.smin = 0.15
        dia.smax = 0.5 
    
        # Add a figure legend
        if True:
            leg2 = fig.legend(dia.samplePoints,
                       [ p.get_label() for p in dia.samplePoints ],
                       numpoints=1, prop=dict(size='small'), loc='upper right', ncol=1)
         
            frame = leg2.get_frame()
            frame.set_edgecolor('None')
            frame.set_facecolor('None')
        plt.title('COOPS data', position=(0.1, 1.04))
        plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
              wspace=wspace1, hspace=hspace1)
        plt.savefig(out_dir + '/taylor_coops' + '.png', dpi=dpi)
        plt.close('all')
    
    
        print ('plot stats on fig ..')
        params = ['cor', 'r2', 'rmse', 'rbias', 'bias', 'mae', 'peak', 'ia', 'skill']
        params = ['cor', 'rmse', 'rbias', 'bias', 'peak', 'ia']
        nall = len(params)
    
        icol = 2
        irow = nall // icol
        params = params[:icol * irow]
        # figure
        fwidth, fheight = ps.get_figsize(300)
        fwidth = fwidth * icol 
        fheight = fheight * irow  
    
        fig, axgrid = plt.subplots(nrows=irow, ncols=icol, sharex=True, sharey=False,
          figsize=(fwidth, fheight),
          facecolor='w', edgecolor='k'
          # ,gridspec_kw = {'width_ratios':[1, 1]} 
          )
    
        axgrid = np.array(axgrid)
        axgrid = axgrid.reshape(icol * irow)
    
        nn = -1
        colors = plt.matplotlib.cm.jet(np.linspace(0, 1, len(base_info.cases.keys())))
    
    
        for param in params:
            nn += 1
            ax = axgrid[nn]
            imodel = 0
            samplePoints = []
            labs = []
            keys = np.sort(base_info.cases.keys())
            for key in keys:
                print (key)
                
                marker = ps.marker[imodel]
                l = ax.plot(imodel, np.abs(base_info.cases[key]['stats'][param]), marker=marker, ls='', c=ps.colors[imodel],
                         markersize=markersize, label=base_info.cases[key]['label'])
                
                labs.append(base_info.cases[key]['label'][9:-6])
                
                samplePoints.append(l)
                imodel += 1
            ylab = string.capitalize ('abs(' + param + ')')  
            ax.set_ylabel(ylab)
            # plt.setp( ax, 'xticklabels', [] )
            
            ax.xaxis.set_ticks(ticks=range(imodel)) 
            ax.xaxis.set_ticklabels(ticklabels=labs)  # ,fontsize=18)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_rotation(90)
            
            ax.set_xlim(-0.1, imodel + 0.1)
    
        if False:
            leg2 = ax.legend(numpoints=1, prop=dict(size='small'), loc='upper right', ncol=1)
            frame = leg2.get_frame()
            frame.set_edgecolor('None')
            frame.set_facecolor('None')
                
    
        plt.subplots_adjust(left=1.5 * left1, bottom=1.4 * bottom1, right=right1, top=1.4 * top1,
              wspace=3 * wspace1, hspace=hspace1)
        plt.savefig(out_dir + '/all_stat_time_series_coops' + '.png', dpi=dpi)
        plt.close('all')
    
    
    if pandas_plots_time_series:
        ####
        ext_min = 2.2
        ext_max = 5
        
        keys = np.sort(base_info.cases.keys())
        for key in keys:
            model = np.array(base_info.cases[key]['model_on_data']).squeeze()
            tmp = pd.DataFrame(model, columns=[base_info.cases[key]['label']])
            if 'all_x' not in globals():
                data = np.array(base_info.cases[key]['data_on_data' ]).squeeze()
                all_x = pd.DataFrame(data, columns=['data'])   
            else:
               all_x = pd.concat([all_x, tmp], axis=1)   
    
        all_x = all_x.dropna()
    
        # pandas plot
        x1 = [-10, 10]
        y1 = [-10, 10]
    
        ############
        # left1   = 0.1    # the left side of the subplots of the figure
        # right1  = 0.9
        # bottom1 = 0.45    # the bottom of the subplots of the figure   (ntr==16   bottom=0.05)
        # top1    = 0.9
        # wspace1 = 0.3
        # hspace1 = 0.3
        ############
        #####
        plt.close('all')
        fig = plt.figure(1, figsize=(6, 7))
        sns.boxplot(data=all_x, whis=2)
        ax = plt.gca()
        # ax.set_ylim(ext_min*0.75,ext_max)
        ax.set_ylabel('Elev. [m]')
        plt.xticks(rotation=90)
        plt.subplots_adjust(left=left1, bottom=2.2 * bottom1, right=right1, top=top1,
                            wspace=1 * wspace1, hspace=1 * hspace1)
    
        plt.savefig(out_dir + '/' + '0box_plot_' + str(ext_min) + '.png', dpi=dpi)              
        plt.close()
        # from pandas import scatter_matrix
        # scatter_matrix(all_x)
    
        ######
        plt.figure()
        for y in all_x.columns[1:]:
            reg = sns.regplot(x=y , y='data', data=all_x, x_estimator=np.mean, label=y)
    
        ax = plt.gca()
        ax.set_ylim(ext_min, ext_max)
        ax.set_xlim(ext_min, ext_max)
        ax.plot(x1, y1, 'k', lw=0.5)
        ax.set_xlabel('models')
        ax.set_aspect(1)
        ax.legend()
        plt.savefig(out_dir + '/' + '0reg_plot_' + str(ext_min) + '.png', dpi=dpi)            
        plt.close()
        ######
        plt.figure()
        for y in all_x.columns[1:]:
            j = sns.jointplot(x=y, y='data', data=all_x, kind='reg') 
            j.ax_joint.set_ylim(ext_min, ext_max)
            j.ax_joint.set_xlim(ext_min, ext_max)
            j.ax_joint.plot(x1, y1, 'k', lw=0.5)
            j.savefig(out_dir + '/' + '0joint_plot_' + str(ext_min) + y + '.png', dpi=dpi)    
            plt.close()      
    
        plt.figure()
        g = sns.pairplot(all_x, x_vars=all_x.columns[1:], y_vars=['data'], kind='reg', size=5, aspect=1)
        for ax in np.array(g.axes).squeeze():
            ax.set_ylim(ext_min, ext_max)
            ax.set_xlim(ext_min, ext_max)
            ax.plot(x1, y1, 'k', lw=0.5)
    
        plt.savefig(out_dir + '/' + '0pair_plot_' + str(ext_min) + '.png', dpi=dpi)          
        plt.close('all')      
         
    
        all_x_diff = all_x * 1.0
    
        for col in all_x.columns:
            all_x_diff[col] = all_x[col] - all_x['data']
    
        t = all_x_diff.describe()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pandas_plots_extreme = False
    if pandas_plots_extreme:
        ####
        ext_min = 2.2
        ext_max = 5
    
        if 'all_extrems_final' in globals():
           del all_extrems_final 
    
        for st_key in all_stations.keys():
            print (st_key)
            stations = all_stations[st_key]
            obs = pd.DataFrame(stations['cwlev']['values'], columns=['data'])
            se = pd.Series(stations['cwlev']['dates'])
            obs = obs.set_index (se)   
            obs.dropna()
            obsh = obs.resample('H').mean()
            # obsh = obs
            
            keys = base_info.cases.keys()
            for key in keys:
                label = base_info.cases[key]['label']
                df = pd.DataFrame(stations[key]['values'], columns=[label])
                df = df.set_index (stations[key]['dates'])
                dfh = df.resample('H').mean()
                # dfh   = df
                obsh = pd.concat([obsh, dfh], axis=1, join_axes=[obsh.index])
            
            if len(obs_ext) > 0:
                ind_max = obs_ext.data.argmax()
                ind_i = np.array(np.where(obs_ext.index == ind_max)).squeeze().item()
                extrems = obs_ext.iloc[ind_i - 6:ind_i + 6]  # scan +- 2 hours for maximas
                extrems = extrems.fillna(extrems.max())  # fill nans with max
                arr = extrems.values
                arr.sort(axis=0)  # sort columns independently
                extrems_sorted = pd.DataFrame(arr, columns=extrems.columns)
                extrems_final = extrems_sorted.iloc[-10:]  # choose 3 max ones
                
                if 'all_extrems_final' not in globals():
                    all_extrems_final = extrems_final
                else:
                    all_extrems_final = pd.concat([all_extrems_final, extrems_final], axis=0)
    
    
        all_extrems_final = all_extrems_final.dropna()
    
        # if 'IKE_HWRF_GFS05d_OC_HSOFS' in all_extrems_final.columns:
        #    all_extrems_final.drop('IKE_HWRF_GFS05d_OC_HSOFS',axis=1,inplace=True)
    
    
        arr = all_extrems_final.values
        arr.sort(axis=0) 
        all_x = pd.DataFrame(arr[-20:, :], columns=all_extrems_final.columns)
    
    
        # pandas plot
        x1 = [-10, 10]
        y1 = [-10, 10]
    
        ############
        # left1   = 0.1    # the left side of the subplots of the figure
        # right1  = 0.9
        # bottom1 = 0.45    # the bottom of the subplots of the figure   (ntr==16   bottom=0.05)
        # top1    = 0.9
        # wspace1 = 0.3
        # hspace1 = 0.3
        ############
        #####
        plt.close('all')
        fig = plt.figure(1, figsize=(6, 7))
        sns.boxplot(data=all_x, whis=2)
        ax = plt.gca()
        # ax.set_ylim(ext_min*0.75,ext_max)
        ax.set_ylabel('Elev. [m]')
        plt.xticks(rotation=90)
        plt.subplots_adjust(left=left1, bottom=2.2 * bottom1, right=right1, top=top1,
                            wspace=1 * wspace1, hspace=1 * hspace1)
    
        plt.savefig(out_dir + '/' + '0box_plot_' + str(ext_min) + '.png', dpi=dpi)              
        plt.close()
        # from pandas import scatter_matrix
        # scatter_matrix(all_x)
    
        ######
        plt.figure()
        for y in all_x.columns[1:]:
            reg = sns.regplot(x=y , y='data', data=all_x, x_estimator=np.mean, label=y)
    
        ax = plt.gca()
        ax.set_ylim(ext_min, ext_max)
        ax.set_xlim(ext_min, ext_max)
        ax.plot(x1, y1, 'k', lw=0.5)
        ax.set_xlabel('models')
        ax.set_aspect(1)
        ax.legend()
        plt.savefig(out_dir + '/' + '0reg_plot_' + str(ext_min) + '.png', dpi=dpi)            
        plt.close()
        ######
        plt.figure()
        for y in all_x.columns[1:]:
            j = sns.jointplot(x=y, y='data', data=all_x, kind='reg') 
            j.ax_joint.set_ylim(ext_min, ext_max)
            j.ax_joint.set_xlim(ext_min, ext_max)
            j.ax_joint.plot(x1, y1, 'k', lw=0.5)
            j.savefig(out_dir + '/' + '0joint_plot_' + str(ext_min) + y + '.png', dpi=dpi)    
            plt.close()      
    
        plt.figure()
        g = sns.pairplot(all_x, x_vars=all_x.columns[1:], y_vars=['data'], kind='reg', size=5, aspect=1)
        for ax in np.array(g.axes).squeeze():
            ax.set_ylim(ext_min, ext_max)
            ax.set_xlim(ext_min, ext_max)
            ax.plot(x1, y1, 'k', lw=0.5)
    
        plt.savefig(out_dir + '/' + '0pair_plot_' + str(ext_min) + '.png', dpi=dpi)          
        plt.close('all')      
         
    
        all_x_diff = all_x * 1.0
    
        for col in all_x.columns:
            all_x_diff[col] = all_x[col] - all_x['data']
    
        t = all_x_diff.describe()
    
    
    
    
    
    
    
    
    
    

"""
##################################################################




def remove_black_list(list_inp, black_list):
    tmp = []
    for item in list_inp:
        if item not in black_list:
            tmp.append(item)
    return tmp


def keep_list(list_inp, all_list):
    tmp = []
    for item in list_inp:
        for al in all_list:
            if item in al:
                tmp.append(item)
    return tmp

def return_matched_items(list1,list2):
    list = []
    for item in list1:
        if item in list2: list.append(item)
    
    return list


def get_station_ids(fort61):
    # fort61 = '/scratch4/COASTAL/coastal/noscrub/Yuji.Funakoshi/coastal_act/2008_ike/hindcast//tide//fort.61.nc'
    fort61_sta_ids = []
    nc2 = netCDF4.Dataset(fort61)
    ncv2 = nc2.variables 
    sta_nams = ncv2['station_name'][:]
    for sta in sta_nams:
        fort61_sta_ids.append(string.join(sta).replace(' ', ''))
    
    nc2.close()
    return fort61_sta_ids

def prepare_ctl_file(fort61, ctlfile=None):
    if ctlfile is None:
       ctlfile = 'ike_wl_stations_all.ctl'

    #defs['lim']['xmin'] = -94. - 10 
    #defs['lim']['xmax'] = -94. + 40 
    #defs['lim']['ymin'] = 29. - 20
    #defs['lim']['ymax'] = 29. + 40

    ind_sta_all, ncv0, stationIDs = get_station_list(fort61, defs['lim'])
    
    fct = open (ctlfile, 'w')
    for ista in ind_sta_all:
        stationID = string.join(ncv0['station_name'][ista]).replace(' ', '')
        sta_pik = base_info.base_dir_coops_piks + '/Ike_coops_stations/sta_pickles/' + stationID + '.pik'
        f2 = file(sta_pik , 'r')
        stations = pickle.load(f2)    
        f2.close()  
        
        line1 = stationID + '  S' + stationID + '   "' + stations['info']['name'] + ', ' + stations['info']['state'] + '" \n'
        line2 = str(stations['info']['lat']) + '  ' + str(stations['info']['lon']) + '   0.0  0.0  0.0 \n'
        print line1 + line2
        fct.write(line1 + line2)
    fct.close()




############################################################
#try:
########### Read SSH data
# read fort61 from model:
print ( 'Read SSH model results > ')
fort61 = base_info.cases[base_info.key0]['dir'] + '/fort.61.nc'
mod , mod_table = get_station_ssh(fort61)

start_dt = max (mod[0].index.min(),ssh[0].index.min()).to_datetime()
end_dt   = min (mod[0].index.max(),ssh[0].index.max()).to_datetime()
 


"""




