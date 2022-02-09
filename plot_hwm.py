#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read hwm data


diff -> 'Add this differential if going from NAVD88 to MSL assuming a positive upwards z coordinate system'
navd88 + diff = msl

#  Delta or convert2msl is always for going from vertical datum to msl by an addition to that datum
# obs_datum = model_datam + convert_datum


"""
__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2017, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"

import os,sys

from   pynmd.plotting.vars_param import *
from   pynmd.plotting import plot_routines as pr
from   pynmd.plotting import plot_settings as ps
from   pynmd.plotting import colormaps as cmaps
from   pynmd.models.adcirc.post import adcirc_post as adcp
from   pynmd.models.tools.unstructured import areaInt
from   pynmd.tools.compute_statistics import find_nearest1d,statatistics

import time
from scipy import stats
from scipy.interpolate import NearestNDInterpolator
from geo_regions import get_region_extent
from atcf import readTrack
import matplotlib.pyplot as plt
import matplotlib.tri as Tri
import numpy as np
import datetime
import string
import glob
import string
import pandas as pd
import netCDF4 as n4
import seaborn as sns
import cmocean

#sns.set_style(style='dark')
sns.set_style(style='ticks')

try:
    os.system('rm base_info.pyc'  )
except:
    pass
if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
import base_info

include_bias = False

curr_time  = time.strftime("%Y%m%d_h%H_m%M_s%S")
#====== subplot adjustments ===============
left1  = 0.1   # the left side of the subplots of the figure
right1 = 0.9   # the right side of the subplots of the figure
bottom1= 0.15    # the bottom of the subplots of the figure   (ntr==16   bottom=0.05)
top1   = 0.9      # the top of the subplots of the figure
wspace1= 0.1   # the amount of width reserved for blank space between subplots
hspace1= 0.15   # the amount of height reserved for white space between subplots
##########################################################################
dpi = 600
ftype = '.png'
Rearth = 6378.1 #[km]
#ftype = '.pdf'

for x in base_info.cases[base_info.key]['dir'].split('/'):
    if 'rt_' in x:  
        prefix = x
    else:
        prefix = '_'.join(base_info.cases[base_info.key0]['dir'].split('/')[-3:])

prefix = 'hwm_' + prefix[0:-1]
#prefix = prefix[0:-1]
out_dir = base_info.out_dir + prefix + '/' #'_' + curr_time + '/' #'/' curr_time+ '/'
os.system('mkdir -p ' + out_dir)
print (' > Output folder: \n  >      ',out_dir)

####################


def find_hwm_v01(xgrd,ygrd,maxe,xhwm,yhwm,elev_hwm,convert_datum=None,bias_cor=None ,flag='pos'):
    from pynmd.tools.compute_statistics import find_nearest1d

    
    """
    In: xgrd,ygrd,maxele: model infos
        xhwm,yhwm,elev_hwm:data infos
        flag: how to treat data model comparison
        flag = all :    find nearset grid point
             = valid:   find nearset grid point with non-nan value
             = pos:     find nearset grid point with positive value
             = neg:     find nearset grid point with negative value
        
        
    Retun: model and data vector
    
    
    #  Delta or convert_datum is always for going from model datum to observed datum by an addition to that datum
    # obs_datum = model_datum + convert_datum

    
    """
    if   flag == 'valid':
        maxe = np.ma.masked_where(maxe==elev_max.fill_value, maxe)
        mask =  maxe.mask   
    elif flag == 'pos':
        mask = [maxe  < 0.0]
    elif flag == 'neg':
        mask = [maxe  > 0.0]
    elif flag == 'all':
        mask = np.isnan(xgrd) 

    else:
        print ('Choose a valid flag > ')
        print ('flag = all :    find nearset grid point ')
        print ('     = valid:   find nearset grid point with non-nan value')
        print ('     = pos:     find nearset grid point with positive value')
        print ('     = neg:     find nearset grid point with negative valueChoose a valid flag > ')
        sys.exit('ERROR') 
    
    mask  = np.array(mask).squeeze()

    xgrd = xgrd[~mask]
    ygrd = ygrd[~mask]
    maxe = maxe[~mask]
    #
    if convert_datum is not None:
        convert_datum = convert_datum[~mask]
    else:
        convert_datum = np.zeros_like(xgrd)
    #
    if bias_cor is not None:
        bias_cor = bias_cor[~mask]
    else:
        bias_cor = np.zeros_like(xgrd)  

    data  = []
    model = [] 
    prox  = []
    xmodel = []
    ymodel = []
    
    for ip in range(len(xhwm)):
        i,dist  = find_nearest1d(xvec = xgrd,yvec = ygrd,xp = xhwm[ip],yp = yhwm[ip])
        data.append (elev_hwm [ip] - convert_datum[i])
        model.append(maxe[i] + bias_cor[i])
        xmodel.append(xgrd[i].item())
        ymodel.append(ygrd[i].item())
        
        prox.append(dist)
    
    data   = np.array(data ).squeeze()
    model  = np.array(model).squeeze()
    prox   = np.array(prox ).squeeze()
    xmodel = np.array(xmodel).squeeze()
    ymodel = np.array(ymodel).squeeze()
    
    return data,xhwm,yhwm,model,xmodel,ymodel,prox

def datetime64todatetime(dt):
    tmp = []
    for it in range(len(dt)):
        tmp.append(pd.Timestamp(dt[it]).to_pydatetime())
    return np.array(tmp)

def plot_track(ax,track,date=None,color = 'r'):
    if date is not None:
        dates = np.array(track['dates'])
        ind = find_nearest_time(dates,date)
        
        ax.plot(track['lon'][ind],track['lat'][ind],'ro',alpha=1,ms=8)
    
    ax.plot(track['lon'],track['lat'],lw=3,color=color,ls='dashed',alpha=1)

def read_track(fname=None):
    
    track = readTrack(fname)
    keys = ['dates', 'lon', 'vmax', 'lat']
    for key in keys:
        tmp   = pd.DataFrame(track[key],columns=[key])

        if 'trc' not in locals():
            trc = tmp
        else:
            trc  = pd.concat([trc,tmp],axis=1)    
   
    trc = trc.drop_duplicates(subset='dates',keep='first')
    trc = trc.set_index (trc.dates)
    trc.drop(columns='dates',inplace=True)
    trc = trc.resample('H').interpolate()
    
    dates = datetime64todatetime(trc.index)
    
    return dict(dates=dates,lon=trc.lon.values, lat=trc.lat.values)



print ('\n\n\n Storm:    ', base_info.name,'\n\n\n')

# Read hwm from csv file
df = pd.read_csv(base_info.hwm_fname)
lon_hwm = df.longitude.values
lat_hwm = df.latitude.values
hwm     = df.elev_m.values
#
taylor_data = {}
stats_data  = {}
model_data  = {}
#
nn = 0

# Read station offset 
df = pd.read_csv(base_info.reference_station['fname'])
# get time leading up to hurricane for background value
tmask = pd.to_datetime(df.date_time).values <= np.datetime64(base_info.tim_lim['xmin'])
offset = (df[tmask]['water_surface_height_above_reference_datum (m)'].mean() - 
    base_info.reference_station['msl_offset']) 

# Read datum conversion info
datumf = n4.Dataset(base_info.datum_fname,'r')
datumlon   = datumf.variables['x'][:]
datumlat   = datumf.variables['y'][:]
datumz     = datumf.variables['depth'][:]
datumf.close()
datum_interp = NearestNDInterpolator(list(zip(datumlon, datumlat)), datumz)

keys = np.sort(list(base_info.cases.keys()))
for key in keys:
    print ('  > ', key)
    maxelevf  =  base_info.cases[key]['dir'] + '/maxele.63.nc'
    ncmaxelev = n4.Dataset(maxelevf,'r')
    elev_max  = ncmaxelev.variables['zeta_max'][:]
    lon       = ncmaxelev.variables['x'][:]
    lat       = ncmaxelev.variables['y'][:]
    depth     = ncmaxelev.variables['depth'][:]
    ncmaxelev.close()
    # add offset
    elev_max = elev_max + offset
    # find datum transform
    msl2navd88 = datum_interp(lon,lat) 

    # read hwm data
    data,xdata,ydata,model,xmodel,ymodel,prox = find_hwm_v01(
        xgrd = lon ,ygrd = lat, maxe = elev_max, xhwm = lon_hwm,
        yhwm = lat_hwm, elev_hwm = hwm , convert_datum = msl2navd88,
        bias_cor = None ,flag = 'valid',
    )

    base_info.cases[key]['data']  = data.squeeze()
    base_info.cases[key]['model'] = model.squeeze()    
    base_info.cases[key]['prox']  = prox.squeeze()    
    base_info.cases[key]['xdata'] = xdata.squeeze()    
    base_info.cases[key]['ydata'] = ydata.squeeze()    
    base_info.cases[key]['xmodel'] = xmodel.squeeze()    
    base_info.cases[key]['ymodel'] = ymodel.squeeze()    

key = list(base_info.cases.keys())[0]
dfa = pd.DataFrame(zip(base_info.cases[key]['data'],
                       base_info.cases[key]['xdata'],
                       base_info.cases[key]['ydata'],
                       base_info.cases[key]['xmodel'],
                       base_info.cases[key]['ymodel']),
                       columns = ['data','xdata','ydata','xmodel','ymodel'] )

####    APPLY RULES to Clean the DATA   ######
for key in np.sort(list(base_info.cases.keys())):
    model = base_info.cases[key]['model'] 
    prox  = base_info.cases[key]['prox'] 
    model[np.abs(model) > 20.0     ]           = np.nan
    if key == base_info.key0: #Only check proximity against first mesh
        model[np.abs(prox)  >  base_info.prox_max] = np.nan

    tmp = pd.DataFrame(model,  columns = [base_info.cases[key]['label']] )
    dfa = pd.concat([dfa,tmp],axis=1)    

# Keep data point where all models have results (joint data points)
dfa = dfa.dropna()

#use cleaned data and coordinates
data   = dfa['data'  ].values
xdata  = dfa['xdata' ].values
ydata  = dfa['ydata' ].values
xmodel = dfa['xmodel'].values
ymodel = dfa['ymodel'].values

## 
dfa_orig  = dfa.copy(deep=True)    
dfa.drop('xdata' ,axis=1,inplace=True)
dfa.drop('ydata' ,axis=1,inplace=True)
dfa.drop('xmodel',axis=1,inplace=True)
dfa.drop('ymodel',axis=1,inplace=True)    


#define model keys
model_keys = []
for key in dfa.columns:
    if key != 'data':
        model_keys.append(key)
   
    if 'tide' in key:
        print ('>>  \n\n\n >> Possible ERROR due to including > only tide <  case in base_info.py \n\n\n')

#############
if include_bias:
    print ('[info:] BIAS Correction included ..')
    bias_all = 0.0
    for key in model_keys:
        bias_all = bias_all + (dfa[key].mean() - dfa['data'].mean())
    
    bias_all = bias_all / len(model_keys)
    
    print ('bias_all=', bias_all)
    for key in model_keys:
        dfa[key] = dfa[key] - bias_all


print ('[info:] plot scatter')
nall = len(model_keys)
icol = min(nall,2)
irow = nall // icol
fwidth,fheight= ps.get_figsize(300)
fwidth  = fwidth  *  icol 
fheight = fheight *  irow  * 1.6

fig,axgrid = plt.subplots(nrows=irow, ncols=icol, sharex=True, sharey=True,
    figsize=(fwidth, fheight),
    facecolor='w', edgecolor='k'
    )

axgrid = np.array(axgrid)
axgrid = axgrid.reshape(icol*irow)   

nn = 0
for key in model_keys:
    print ('  > ', key)
    
    model = dfa[key]
    ax  = axgrid[nn]
    #
    pr.plot_scatter(ax, data, model.values , var=defs['elev'], color='b', nn=nn, title=key)
    #
    stats_data[key]    = statatistics (data,model.values)
    taylor_data.update({key:[model.std(ddof=1), np.corrcoef(data, model)[0,1]]})
    
    pattern_rms_diff = np.std(model-data) 
    #np.sqrt((((data-data.mean()) - (model-model.mean()))**2).mean())
    
    print ('   > N= ', len(model))
    print ('   > RMS= ', pattern_rms_diff)
    
    nn += 1

plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
      wspace=0.7*wspace1, hspace=1.1 * hspace1)
plt.savefig(out_dir+ '/scatters_HWM' + ftype,dpi=dpi)
plt.close('all')

###################################################################

print ('Plot map for HWM data ...')
track = read_track(fname = base_info.track_fname)
##

keys = list(base_info.cases.keys())
for kk,key in enumerate(keys):
    print ('  > ', key)
    maxelevf  =  base_info.cases[key]['dir'] + '/maxele.63.nc'
    ncmaxelev = n4.Dataset(maxelevf,'r')
    elev_max  = ncmaxelev.variables['zeta_max'][:]
    lon       = ncmaxelev.variables['x'][:]
    lat       = ncmaxelev.variables['y'][:]
    depth     = ncmaxelev.variables['depth'][:]
    elems     = ncmaxelev.variables['element'][:,:]-1  # Move to 0-indexing by subtracting 1
    mask      = elev_max.mask[elems].any(axis=1)
    #mask      = elev_max < 0
    tri = Tri.Triangulation(lon, lat, triangles=elems, mask=mask)
 
    # add offset
    elev_max = elev_max + offset
    # apply mask     
    #mask      = elev_max < 0
    #elev_max[elev_max.mask] = 0.0
    #val       = elev_max
    if key == base_info.key0: 
        var  = defs['elev']
        tick_nos = 6
        val_interp0 = NearestNDInterpolator(list(zip(lon, lat)), elev_max)
    else:
        var  = defs['elev_diff']
        tick_nos = 5
        elev_max0 = val_interp0(lon,lat)
        elev_max  = elev_max - elev_max0
    ncmaxelev.close()
   
    #
    fig, ax = pr.make_map()
    fig.set_size_inches(9,9)
    
    lim = get_region_extent(region = base_info.regions[0])
    extent = [lim['xmin'],lim['xmax'],lim['ymin'],lim['ymax']+0]        
    ax.set_extent(extent)
    
    # compute area of inundation within plot region
    inun_mask       = ((tri.x[elems] >= lim['xmin']).all(axis=1) & 
                      (tri.x[elems] <= lim['xmax']).all(axis=1) & 
                      (tri.y[elems] >= lim['ymin']).all(axis=1) & 
                      (tri.y[elems] <= lim['ymax']).all(axis=1) & 
                      (depth.data[elems] < 0).all(axis=1) & (~mask))
    areas, tot_area = areaInt(
        np.vstack((np.deg2rad(tri.x)*np.cos(np.deg2rad(tri.y)),np.deg2rad(tri.y))).T, 
        tri.triangles[inun_mask],
    )
    stats_data[model_keys[kk]]['inundation area'] = tot_area * Rearth**2 # [km^2]
    
    # Track 
    plot_track(ax,track,date=None)
   
    # Shoreline 
    cond1 = ax.tricontour(tri, depth+0.1 ,levels = [0.0], colors='k', linewidths=0.01, alpha= 0.5)
    
    # Max Ele
    vmin = var['vmin'] 
    vmax = var['vmax'] 
    cmap = var['cmap']
    levels = np.linspace(vmin,vmax,25+1)
    cf1    = ax.tricontourf(tri,elev_max.data,levels=levels,cmap = cmap,extend='both',alpha = 1.0)
    cb     = plt.colorbar(cf1,shrink = 0.5,ticks = np.linspace(vmin,vmax,tick_nos))   
    cb.set_label(var['label'])
    
    # HWM points
    if key == base_info.key0:
        ax.scatter(xdata,ydata,marker='o',s=8,c=data,vmin=vmin,vmax=vmax,cmap=cmap,zorder=3,edgecolors='k',alpha=1.0)
    
    titl = base_info.cases[key]['label'] 
    ax.set_title(titl)
    filename = out_dir + '/maps_hwm_obs_'+titl.replace(' ','_')+ftype
    plt.savefig(filename,dpi=dpi)
    plt.close('all')
 
################
print ('[info:] plot Taylor Dig.')

from pynmd.plotting import taylor
markersize = 6
fig = plt.figure(6,figsize=(9,9))
fig.clf()

refstd = data.std(ddof=1)

# Taylor diagram
dia = taylor.TaylorDiagram(refstd, fig=fig, rect=111, label="Reference")
 
colors = plt.matplotlib.cm.jet(np.linspace(0,1,len(list(taylor_data.keys()))))
 
# Add samples to Taylor diagram
for imodel in range(len(list(taylor_data.keys()))):
    key      = model_keys[imodel]
    stddev   = taylor_data[key][0]
    corrcoef = taylor_data[key][1]
    marker   = ps.marker[imodel]
    dia.add_sample(stddev, corrcoef,ref=False, marker=marker, ls='', c=ps.colors[imodel],
               markersize=markersize,label=key)
     
# add reference point for data     
dia.add_sample(refstd, 1.0 ,ref=True, marker='*', ls='',  c='k',
               markersize=markersize*1.5,label='Ref.')
 
# Add RMS contours, and label them
contours = dia.add_contours(levels=8,data_std=refstd,colors='0.5')
plt.clabel(contours, inline=1,fmt = '%.3g', fontsize=10)
 
# Add a figure legend
leg2=fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints ],
           numpoints=1, prop=dict(size='small'), loc='upper right',
           ncol=2)

frame=leg2.get_frame()
frame.set_edgecolor('None')
frame.set_facecolor('w')

plt.title( 'HWM data' + '  N=' + str(len(model)), position=(0.1, 1.04))
plt.subplots_adjust(left=left1, bottom=bottom1, right= right1, top= top1,
      wspace=wspace1, hspace=hspace1)
plt.savefig(out_dir+ '/taylor_HWM' + ftype,dpi=dpi)
plt.close('all')

#
print ('[info:] plot stats on fig ..')
#params = ['cor', 'r2', 'bias', 'mae', 'rmse', 'skill']
params = ['mae', 'rmse', 'inundation area']
nall = len(params)

icol = 1 #min(nall,2)
irow = nall//icol
params = params[:icol*irow]
#figure
fwidth,fheight= ps.get_figsize(300)
fwidth  = fwidth  *  icol *1.25
fheight = fheight *  irow 

fig,axgrid = plt.subplots(nrows=irow, ncols=icol, sharex=True, sharey=False,
  figsize=(fwidth, fheight),
  facecolor='w', edgecolor='k',
)

axgrid = np.array(axgrid)
axgrid = axgrid.reshape(icol*irow)

nn = -1
colors = plt.matplotlib.cm.jet(np.linspace(0,1,len(list(base_info.cases.keys()))))

for param in params:
    nn += 1
    ax  = axgrid[nn]
    imodel = 0
    samplePoints = []
    labs = []
    #for key in stats_data.keys():  
    for key in model_keys:  
        #print key
        marker   = ps.marker[imodel]
        l = ax.plot(imodel,(stats_data[key][param]),marker=marker, ls='', c=ps.colors[imodel],
                 markersize=markersize,label=key)
        
        #labs.append(key[9:-6])
        labs.append(key[:])
        
        samplePoints.append(l)
        imodel += 1
       
    ylab = param.capitalize()
    if param in 'rmsebiasmae':
        ylab = ylab.upper() + ' [m]'
    if param in 'inundation area':
        ylab = ylab + ' [km$^2$]'

    ax.set_ylabel( ylab)
    ax.locator_params(axis='y', nbins=4)
    ax.xaxis.set_ticks(ticks=range(imodel)) 
    ax.xaxis.set_ticklabels(ticklabels=labs) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_rotation(90)
    
    ax.set_xlim(-0.2,imodel+0.2)
    ax.grid()
        
fig.suptitle('HWM errors (N=' + str(len(model)) + ') and inundation areas')
plt.subplots_adjust(left = 1.5 * left1, bottom = 2.5 * bottom1, right=right1, top=top1,
      wspace=4 * wspace1, hspace=hspace1)
plt.savefig(out_dir+ '/all_stat_HWM' + ftype,dpi=dpi)
plt.close('all')
