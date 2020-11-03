#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot multi axes ####
"""
__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2017, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"


from   pynmd.plotting.vars_param import *
from   pynmd.plotting import plot_routines as pr
import pynmd.models.adcirc.post as adcp 
from   pynmd.tools.tide_analysis import tappy_filters
from   pynmd.plotting import colormaps as cmaps
import pynmd.plotting.plot_settings  as ps
from pynmd.tools.compute_statistics import find_nearest1d
import netCDF4 as n4
from   collections import defaultdict
#import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import datetime,time
import string
#import glob
import time
import pandas as pd
import seaborn as sns
import itertools
from geo_regions import get_region_extent   
import pynmd.plotting.colormaps as cmaps
 

sns.set_style(style='whitegrid')
#sns.set_style(style='dark')
sys.path.append('/home/Saeed.Moghimi/opt/pycodes/csdlpy')

try:
    os.system('rm base_info.pyc')
except:
    pass
if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
import base_info

tim_lim = base_info.tim_lim

plot_timeseries = True
pandas_plots = True
plot_taylor = True
plot_rtofs = False
sub_tidal = False
 
############################
curr_time = time.strftime("%Y%m%d_h%H_m%M_s%S")
fs = 12
fs2 = 10       

# plt.rc('font', family='serif')
plt.rcParams.update(\
        {'axes.labelsize': fs,
         'text.fontsize': fs,
         'xtick.labelsize': fs,
         'ytick.labelsize': fs,
         'axes.titlesize': fs,
         'legend.fontsize': fs / 1.5,
         })


sys.path.append('/home/Saeed.Moghimi/opt/pycodes/csdlpy/')
import adcirc
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
        fname = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/tracks/ike_bal092008.dat'
    
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



def get_diff(varname,ind0,ind1):
    fname = base_info.cases[base_info.key0]['dir'] + defs[varname]['fname']
    nc0   = n4.Dataset(fname)
    ncv0  = nc0.variables
    val0  = ncv0[defs[varname]['var']][ind0]
    #
    fname = base_info.cases[base_info.key1]['dir'] + defs[varname]['fname']
    nc1   = n4.Dataset(fname)
    ncv1  = nc1.variables
    val1  = ncv1[defs[varname]['var']][ind1]

    return (val1 - val0)
    
def get_val(varname,ind1=None):
    #
    
    fname = base_info.cases[base_info.key1]['dir'] + defs[varname]['fname']
    nc1   = n4.Dataset(fname)
    ncv1  = nc1.variables
    if ind1 is None:
        val1  = ncv1[defs[varname]['var']][:]
    else:
        val1  = ncv1[defs[varname]['var']][ind1]
    
    nc1.close()
    return val1


#### END of Funcs



for x in base_info.cases[base_info.key0]['dir'].split('/'):
    if 'rt_' in x:  prefix = x


t1 = tim_lim['xmin'].isoformat()[2:13] + '_' + tim_lim['xmax'].isoformat()[2:13] + '____MAAAAAAXXXXXXX_' + curr_time

out_dir = base_info.out_dir + prefix + t1 + '/'
# out dir and scr back up
scr_dir = out_dir + '/scr/'
os.system('mkdir -p ' + scr_dir)
args = sys.argv
scr_name = args[0]
os.system('cp -fr  ' + scr_name + '    ' + scr_dir)
os.system('cp -fr  base_info.py     ' + scr_dir)
print out_dir
####################
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

#populate recording points
# read base_run only_tide time info
fname0  = base_info.cases[base_info.key0]['dir'] + '/fort.63.nc'
nc0    = n4.Dataset(fname0)
ncv0   = nc0.variables
depth  = ncv0['zeta'][:]
dep    = ncv0['depth'][:]
dates0 = n4.num2date(ncv0['time'][:],ncv0['time'].units)
indd0   = np.array(np.where((dates0 > base_info.tim_lim['xmin'] )&(dates0<base_info.tim_lim['xmax'])  )).squeeze()
nc0.close()
#
fname1  = base_info.cases[base_info.key1]['dir'] + '/fort.63.nc'
nc1     = n4.Dataset(fname1)
ncv1    = nc1.variables
dates1 = n4.num2date(ncv1['time'][:],ncv1['time'].units) 
indd1   = np.array(np.where((dates1 > base_info.tim_lim['xmin'] )&(dates1<base_info.tim_lim['xmax'])  )).squeeze()
nc1.close()
#
#construct HSOFS Tri mask
lon,lat,tri  = adcp.ReadTri(base_info.cases[base_info.key0]['dir'])
#                    

max_locs = defaultdict(dict)
base_info.varnames = ['elev']
print base_info.cases[base_info.key1]['dir']


print 'region > ', base_info.varnames

Finx_max_on_each_time_step = False

if Finx_max_on_each_time_step:
    for varname in base_info.varnames:   
        for ind1 in indd1[::1][:]:
            print '-------------------------------------------'
            for region in base_info.regions:
                lim = get_region_extent(region = region)
                date  = dates1[ind1]
                # find matching index for new netcdf file 
        
                print ind1, region ,varname, date
                
                try:
                    pres =  get_val('pmsl',ind1)
                except:
                    pres = None
                    pass   
                
                if varname in ['elev','u','v']:
                    ind0  = np.array(np.where((dates0== date))).squeeze().item()
                    val   = get_diff(varname,ind0,ind1)
    
                
                if varname in ['rad']:
                    radx = 'radstress_x'
                    rady = 'radstress_y'
                    #
                    defs[varname]['var'] = radx
                    valx = get_val(varname,ind1)
                    
                    defs[varname]['var'] = rady
                    valy = get_val(varname,ind1)
                    
                    val = np.sqrt(valx*valx+valy*valy)
                    defs['rad']['vmin']  =  0.0
                    defs['rad']['vmax']  =  1e-3
                                   
                if varname in ['wind']:
                    radx = 'windx'
                    rady = 'windy'
                    #
                    defs[varname]['var'] = radx
                    valx = get_val(varname,ind1)
                    
                    defs[varname]['var'] = rady
                    valy = get_val(varname,ind1)
                    
                    val = np.sqrt(valx*valx+valy*valy)
                    defs['wind']['vmin']  =  0.0
                    defs['wind']['vmax']  =  1e-3
        
                if varname in ['wind_']:
                    radx = 'windx'
                    rady = 'windy'
                    #
                    defs[varname]['var'] = radx
                    valx = get_val(varname,ind1)
                    
                    defs[varname]['var'] = rady
                    valy = get_val(varname,ind1)
                    
                    val = np.sqrt(valx*valx+valy*valy)
                    defs['wind']['vmin']  =  0.0
                    defs['wind']['vmax']  =  2e-4                
                
                val[np.isnan(val)] = 0.0
    
                [ind_box] = np.where((lon > lim['xmin']) & (lon < lim['xmax']) &(lat > lim['ymin']) & (lat < lim['ymax']))
                
                
                ind_max = np.argmax(val[ind_box])
                lon_max = lon[ind_box][ind_max]
                lat_max = lat[ind_box][ind_max]
                val_max = val[ind_box][ind_max]
                [i],prox = find_nearest1d(xvec = lon,yvec = lat,xp = lon_max,yp = lat_max)
                    
                if np.abs(val_max) > 0.2:
                    max_locs[varname+str(i)] = dict( ind = i,  lon = lon_max, lat = lat_max, val= val_max)
else:
    varname = base_info.varnames[0]
    fname = base_info.cases[base_info.key0]['dir'] + 'maxele.63.nc'
    nc0   = n4.Dataset(fname)
    ncv0  = nc0.variables
    val0  = ncv0['zeta_max'][:]
    #
    fname = base_info.cases[base_info.key1]['dir'] + 'maxele.63.nc'
    nc1   = n4.Dataset(fname)
    ncv1  = nc1.variables
    val1  = ncv1['zeta_max'][:]
    val_dif =  np.abs(val1 - val0)
    val_dif[val_dif.mask] = 0.0
    #
    val = val1
    val[val.mask] = 0.0    
    #
    #
    nmax = 50000
    jp   = 250
    
    #if True:
    #    ind_max = np.where( (val1 > 5 ) & ( np.abs(val_dif) > 0.2 ) )
    #    lon_maxa = lon [ind_max][:nmax:jp]
    #    lat_maxa = lat [ind_max][:nmax:jp]
    #    val_maxa = val [ind_max][:nmax:jp]
    #    dep_maxa = dep [ind_max][:nmax:jp]
    #else:
    lim = get_region_extent(region = base_info.regions[0])
    [ind_box] = np.where((lon > lim['xmin']) & (lon < lim['xmax']) &(lat > lim['ymin']) & (lat < lim['ymax']))
    
    if True:
        ind_max = np.argsort(a=val[ind_box])
        ind_max = ind_max[::-1]
    else:
        ind_max = np.where( (val1[ind_box] > 5 ) & ( np.abs(val_dif[ind_box]) > 0.2 ) )

    lon_maxa = lon[ind_box] [ind_max][:nmax:jp]
    lat_maxa = lat[ind_box] [ind_max][:nmax:jp]
    val_maxa = val[ind_box] [ind_max][:nmax:jp]
    dep_maxa = dep[ind_box] [ind_max][:nmax:jp]
    #sys.exit()
    
    
    nmax = len(lon_maxa)

    for il in range(len(lon_maxa)):
        lon_max  = lon_maxa [il]
        lat_max  = lat_maxa [il]
        val_max  = val_maxa [il]
        dep_max  = dep_maxa [il]
        [i],prox = find_nearest1d(xvec = lon,yvec = lat,xp = lon_max,yp = lat_max)
        max_locs[varname+str(i)] = dict( ind = i,  lon = lon_max, lat = lat_max, val= val_max, dep = dep_max)

    
sta_list_all = np.sort(max_locs.keys())

params = ['elev']
icol = 1
irow_max = 1

nplot = len(sta_list_all)// (irow_max * icol)
iloc = 0
for iplot in range(nplot):
    if len(sta_list_all) < (irow_max * icol):
        print ' ... All list'
        sta_list = sta_list_all
        # match the numebr to clumns
        sta_list = sta_list[:n]
    else:
        print ' ... partial list'

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
    for ista in range(len(sta_list)):
        stationID = sta_list[ista]
        print stationID
        
        header = 'At Lon= ' + str(max_locs[stationID]['lon'])[:8]+ ' Lat= ' + str(max_locs[stationID]['lat'])[:8]
        prefix =  header.replace(':','_').replace(' ','_').replace(',','_').replace(',','_')
        
        print  header
        
        # start plotting
        ax = axgrid[nn]
        ic = 0
        
        defs['elev']['vmin'] = -1
        defs['elev']['vmax'] =  1
        cl = itertools.cycle(sns.color_palette(palette='dark', n_colors=2 * len(base_info.cases.keys())))
        # cl = itertools.cycle(sns.color_palette(palette = 'mute', n_colors = 2 * len(base_info.cases.keys()) ) )
        
        keys = np.sort(base_info.cases.keys())
        for key in keys:
                # try:
                print key
                fort63 = base_info.cases[key]['dir'] + '/fort.63.nc'
                nc = n4.Dataset(fort63)
                ncv = nc.variables
                tim,uni_indx = np.unique(ncv['time'][:], return_index=True)
                dates = n4.num2date(ncv['time'][uni_indx], ncv['time'].units)
                val = ncv[defs['elev']['var']][uni_indx, max_locs[stationID]['ind']]
                nc.close()
                #
                defs['elev']['vmin'] = min (0,1.1 * val.min() , defs['elev']['vmin'])
                defs['elev']['vmax'] = max (1.1 * val.max() , defs['elev']['vmax'])
                #
                data = dict(xx=dates           ,
                            val=val            ,
                            #
                            var=defs['elev'],
                            lim=tim_lim,
                            )
                #
                args = dict(
                            title=header,
                            label=base_info.cases[key]['label'],
                            color=ps.cl[ic],
                            # color      = next(cl),                          
                            plot_leg=True
                            # panel_num = nn,
                            )
                #
        
                if sub_tidal:
                    print 'sub_tidal'
                    dates1, val1 = do_tappy_tide_analysis(dates, val)
                    data['xx'] = dates1
                    data['val'] = val1
                    pr.TimeSeriesPlot(ax, data, args)
                else: 
                    pr.TimeSeriesPlot(ax, data, args)
              
        
                # ax.plot(dates,val)
                ic += 1
                # except:
                # print '        >      unsucessful key >' , key
                # #sys.exit()
                # pass  
         
        nn += 1         
        ax.plot(dates,-max_locs[stationID]['dep']*np.ones_like(dates),'--',color='k',lw=1,  label='Ground level' )
    #plt.savefig(out_dir + '/' +str(100+iplot)+ 'compare_spin_' + '.png', dpi=dpi) 
    plt.savefig(out_dir + '/' +str(100+iplot)+ '_tim_' +prefix + '_' + '.png', dpi=dpi) 

         
     # plt.show()
    plt.close('all')
    
    
    
    ##################################################################################
    print 'Plot map for recording points ...'
    #construct HSOFS Tri mask
    fname  = base_info.cases[base_info.key1]['dir'] + '/fort.63.nc'
    nc0    = n4.Dataset(fname)
    ncv0   = nc0.variables
    depth  = ncv0['depth'][:]
    depth [depth < -10] = -10
    nc0.close()
    
    
    deltap = 0.5
    fig, ax = adcp.make_map()
    fig.set_size_inches(7,7)
    extent = [defs['lim']['xmin']-deltap,defs['lim']['xmax']+deltap,defs['lim']['ymin']-deltap,defs['lim']['ymax']+deltap]        
    ax.set_extent(extent)

    #track = read_track(fname = base_info.track_fname)
    #plot_track(ax,track,date=None)
    
    track = readTrack(base_info.track_fname)
    ax.plot(track['lon'],track['lat'],'r-',lw=2,alpha=1,ms=10, markevery = 5)#, marker = r'$\bigotimes$',mec='None',mew=2)  #marker = r'$\S$'
    
    cond1 = ax.tricontour(tri,depth+0.1 ,levels=[0.0]  ,colors='k',lw=0.25, alpha= 0.5)
    
    
    if False:
        #plot mesh
        ax.triplot(tri, 'w-', lw=0.1, alpha=0.4)


    #print ' tricontourf '
    vmin = 0
    vmax = 6
    dv = (vmax-vmin)/50.0
    levels=np.arange(vmin,vmax+dv,dv)
    #val [val < vmin] = vmin+1e-3
    cmap = cmaps.jetMinWi
    cf1 = ax.tricontourf(tri,val1,levels=levels,cmap=cmap, extend='both')#extend='max' )  #,extend='both'
    
    
    
    
    # set scale for vertical vector plots
    pos_ax   = np.array (ax.get_position ())
    aheight  = pos_ax[1][1] -pos_ax[0][1]
    awidth   = pos_ax[1][0] -pos_ax[0][0]
    #only for vector plot
    fig = plt.gcf()
    #fwidth  = fig.get_size_inches()[0]
    #fheight = fig.get_size_inches()[1]
    fwidth   = 1
    fheight  = 1
    wscale  = aheight*fheight/(awidth*fwidth) * (lim['xmax']-lim['xmin'])/(lim['ymax']-lim['ymin'])
    #
    xsub1 = pos_ax[0][0]
    xsub2 = pos_ax[1][0]
    ysub1 = pos_ax[0][1]            
    ysub2 = pos_ax[1][1]
    
    cb_dx = 1.05
    cb_dy = 0.3
    cb_size = 0.3
    cbax    = fig.add_axes([xsub1+ cb_dx * (xsub2-xsub1), ysub1 + cb_dy * (ysub2-ysub1), 0.01, cb_size]) 
    
    cb  = plt.colorbar(cf1,cax=cbax,ticks = [vmin,(vmin+vmax)/2,vmax],format='%1.4g',orientation='vertical')      


    
    #plot all stations
    if False:
        for stationID in sta_list_all:
            ax.scatter(max_locs[stationID]['lon'],max_locs[stationID]['lat'],s=15,c='y',zorder=3,edgecolors='None')
    
    #plot these stations
    for stationID in sta_list:
        ax.scatter(max_locs[stationID]['lon'],max_locs[stationID]['lat'],s=50,c='r',zorder=3,marker='s')


    plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
                wspace=wspace1, hspace=hspace1)

     
    #ax.set_title(header)   
    filename = out_dir + '/' +str(100+iplot)+'_map_' +prefix + '_' + '.png'
    
    plt.savefig(filename,dpi=150)
    plt.close('all')


