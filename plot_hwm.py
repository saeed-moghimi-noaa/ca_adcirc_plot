#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read hwm data


diff -> 'Add this differential if going from NAVD88 to MSL assuming a positive upwards z coordinate system'
navd88 + diff = msl


#  Delta or convert2msl is always for going from vertical datum to msl by an addition to that datum
# MSL = Vert_datam + convert2msl 


"""
__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2017, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"

#import netCDF4 as n4
#from   collections import defaultdict
import os,sys

#sys.path.append('/home/moghimis/linux_working/00-working/04-test-adc_plot/')
#sys.path.append('/home/moghimis/linux_working/00-working/04-test-adc_plot/csdlpy')


from   pynmd.plotting.vars_param import *
from   pynmd.plotting import plot_routines as pr
from   pynmd.plotting import plot_settings as ps
from   pynmd.plotting import colormaps as cmaps
from   pynmd.models.adcirc.post import adcirc_post as adcp
from   pynmd.tools.compute_statistics import find_nearest1d,statatistics

import time
from scipy import stats
from geo_regions import get_region_extent
#import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.tri as Tri
import numpy as np
import datetime
import string
import glob
#import time
import string
import pandas as pd
import netCDF4 as n4
import seaborn as sns


#sys.path.append('/scratch2/COASTAL/coastal/save/Saeed.Moghimi/opt/pycodes/csdlpy')
#import adcirc


#sns.set_style(style='dark')
sns.set_style(style='ticks')

try:
    os.system('rm base_info.pyc'  )
except:
    pass
if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
import base_info

pandas_plots = True
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
#ftype = '.pdf'


for x in base_info.cases[base_info.key]['dir'].split('/'):
    if 'rt_' in x:  
        prefix = x
    else:
        prefix = ''.join(base_info.cases[base_info.key0]['dir'].split('/')[-3:])

prefix = 'hwm_' + prefix
out_dir = base_info.out_dir + prefix + curr_time+ '/'
# out dir and scr back up
scr_dir = out_dir + '/scr/'
os.system('mkdir -p ' + scr_dir)
args=sys.argv
scr_name = args[0]
os.system('cp -fr  '+scr_name +'    '+scr_dir)
os.system('cp -fr  *.py             '+scr_dir)
print (' > Output folder: \n  >      ',out_dir)

####################


def find_hwm_v01(xgrd,ygrd,maxe,xhwm,yhwm,elev_hwm,convert2msl=None,bias_cor=None ,flag='pos'):
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
    
    
    #  Delta or convert2msl is always for going from vertical datum to msl by an addition to that datum
    # MSL = Vert_datam + convert2msl 

    
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

        #mask = [maxe < -900.0]

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
    if convert2msl is not None:
        convert2msl = convert2msl[~mask]
    else:
        convert2msl = np.zeros_like(xgrd)
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
        i,pr  = find_nearest1d(xvec = xgrd,yvec = ygrd,xp = xhwm[ip],yp = yhwm[ip])
        data.append (elev_hwm [ip] + convert2msl[i])
        model.append(maxe[i]+bias_cor[i])
        xmodel.append(xgrd[i].item())
        ymodel.append(ygrd[i].item())
        
        prox.append(pr)
    
    data   = np.array(data ).squeeze()
    model  = np.array(model).squeeze()
    prox   = np.array(prox ).squeeze()
    xmodel = np.array(xmodel).squeeze()
    ymodel = np.array(ymodel).squeeze()
    
    
    #
    #maskf = [model < 0.0]
    #maskf  = np.array(maskf).squeeze()
    #return data[~maskf],model[~maskf],prox[~maskf],xhwm[~maskf],yhwm[~maskf]
    return data,xhwm,yhwm,model,xmodel,ymodel,prox


def find_hwm(tri,maxe,xhwm,yhwm,elev_hwm,bias_cor=None,flag='all'):
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
    
    
    #  Delta or convert2msl is always for going from vertical datum to msl by an addition to that datum
    # MSL = Vert_datam + convert2msl 

    
    """
#     if   flag == 'valid':
#         mask = np.isnan(xgrd) 
#     elif flag == 'pos':
#         mask = [maxe  < 0.0]
#     elif flag == 'neg':
#         mask = [maxe  > 0.0]
#     elif flag == 'all':
#         mask =  maxe.mask   
#         #mask = [maxe < -900.0]
# 
#     else:
#         print ('Choose a valid flag > '
#         print ('flag = all :    find nearset grid point '
#         print ('     = valid:   find nearset grid point with non-nan value'
#         print ('     = pos:     find nearset grid point with positive value'
#         print ('     = neg:     find nearset grid point with negative valueChoose a valid flag > '
#         sys.exit('ERROR') 
#     
#     mask  = np.array(mask).squeeze()
    #maxe = elev_max
    #xhwm = lon_hwm
    #yhwm = lat_hwm
    #elev_hwm = hwm
    #
    xgrd = tri.x
    ygrd = tri.y
    #
    if bias_cor is  None:
        bias_cor = np.zeros_like(xgrd)  
    #
    data      = []
    model     = [] 
    model_x   = []
    model_y   = []
    prox      = []
    prox_coef = []
    #
    for ip in range(len(xhwm)):
        i, pr  = find_nearest1d(xvec = xgrd [~maxe.mask],yvec = ygrd[~maxe.mask],xp = xhwm[ip],yp = yhwm[ip])  #valid mesh
        ir,prr = find_nearest1d(xvec = xgrd             ,yvec = ygrd            ,xp = xhwm[ip],yp = yhwm[ip])  #all mesh  
        #if pr > 1.2 * prr:
        #    print pr, prr, pr/prr
        data.append (elev_hwm [ip])
        model.append  (maxe[~maxe.mask][i]+bias_cor[~maxe.mask][i])
        model_x.append(xgrd[~maxe.mask][i])
        model_y.append(ygrd[~maxe.mask][i])
        prox.append(pr)
        prox_coef.append(pr/prr)
    #
    data  = np.array(data ).squeeze()
    model = np.array(model).squeeze()
    prox  = np.array(prox ).squeeze()
    #
    maskf = (np.array(prox_coef) > 10) | (data < 1)
    maskf = np.array(maskf).squeeze()
    return data[~maskf],model[~maskf],prox[~maskf], xhwm[~maskf], yhwm[~maskf]

def datetime64todatetime(dt):
    tmp = []
    for it in range(len(dt)):
        tmp.append(pd.Timestamp(dt[it]).to_pydatetime())
    return np.array(tmp)

def plot_track(ax,track,date=None,color = 'r'):
    if date is not None:
        dates = np.array(track['dates'])
        #ind = np.array(np.where((dates==date))).squeeze().item()
        ind = find_nearest_time(dates,date)
        
        ax.plot(track['lon'][ind],track['lat'][ind],'ro',alpha=1,ms=8)
    
    ax.plot(track['lon'],track['lat'],lw=3,color=color,ls='dashed',alpha=1)


#sys.path.append('/home/Saeed.Moghimi/opt/pycodes/csdlpy/')
#import adcirc
from atcf import readTrack

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



print ('\n\n\n Storm:    ', base_info.name,'\n\n\n')

#for map plot
if False:
    lon,lat,tri  = adcp.ReadTri(base_info.cases[base_info.key1]['dir'])
else:
    fname  =  base_info.cases[base_info.key0]['dir'] + '/maxele.63.nc'
    nc0    = n4.Dataset(fname)
    ncv0   = nc0.variables
    lon   = ncv0['x'][:]
    lat   = ncv0['y'][:]
    elems  = ncv0['element'][:,:]-1  # Move to 0-indexing by subtracting 1
    tri = Tri.Triangulation(lon,lat, triangles=elems)

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

    if True:
        data,xdata,ydata,model,xmodel,ymodel,prox = find_hwm_v01(xgrd = lon ,ygrd = lat, maxe = elev_max, xhwm = lon_hwm,
                                    yhwm = lat_hwm, elev_hwm = hwm , convert2msl = None,
                                    bias_cor = None ,flag='valid')
    
    #else:
    #    data,model,prox,xdata,ydata = find_hwm    (tri = tri, maxe = elev_max, xhwm = lon_hwm, yhwm = lat_hwm, elev_hwm = hwm , bias_cor = None)
    
    #sys.exit()

    base_info.cases[key]['data']  = data.squeeze()
    base_info.cases[key]['model'] = model.squeeze()    
    base_info.cases[key]['prox']  = prox.squeeze()    
    base_info.cases[key]['xdata'] = xdata.squeeze()    
    base_info.cases[key]['ydata'] = ydata.squeeze()    
    base_info.cases[key]['xmodel'] = xmodel.squeeze()    
    base_info.cases[key]['ymodel'] = ymodel.squeeze()    



#sys.exit()
key = list(base_info.cases.keys())[0]
dfa = pd.DataFrame(zip(base_info.cases[key]['data'],
                       base_info.cases[key]['xdata'],
                       base_info.cases[key]['ydata'],
                       base_info.cases[key]['xmodel'],
                       base_info.cases[key]['ymodel']),
                       columns = ['data','xdata','ydata','xmodel','ymodel'] )

#check for nans for the second time (max values)


####    APPLY RULES to Clean the DATA   ######
for key in np.sort(list(base_info.cases.keys())):
    model = base_info.cases[key]['model'] 
    prox  = base_info.cases[key]['prox'] 
    model[np.abs(model) > 20.0     ]           = np.nan
    model[np.abs(prox)  >  base_info.prox_max] = np.nan

    tmp = pd.DataFrame(model,  columns = [base_info.cases[key]['label']] )
    #print tmp.shape
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
icol = 2
irow = nall // icol
#figure
fwidth,fheight= ps.get_figsize(300)
fwidth  = fwidth  *  icol 
fheight = fheight *  irow  * 1.6

fig,axgrid = plt.subplots(nrows=irow, ncols=icol, sharex=True, sharey=True,
    figsize=(fwidth, fheight),
    facecolor='w', edgecolor='k'
    #,gridspec_kw = {'width_ratios':[1, 1]} 
    )

axgrid = np.array(axgrid)
axgrid = axgrid.reshape(icol*irow)   




#remove locations afrom data frame
#      dfa.drop('xdata',axis=1,inplace=True)
#      dfa.drop('ydata',axis=1,inplace=True)
#model_keys = dfa.columns

nn = 0
for key in model_keys:
    print ('  > ', key)
    
    model = dfa[key]
    #nn = 0
    ax  = axgrid[nn]
    #defs['elev']['vmin'] = 0.0
    #defs['elev']['vmax'] = 7.0
    #
    pr.plot_scatter(ax, data, model     , var=defs['elev'],color='b',nn=nn, title=key)
    #
    stats_data[key]    = statatistics (data,model)
    taylor_data.update({key:[model.std(ddof=1), np.corrcoef(data, model)[0,1]]})
    
    pattern_rms_diff = np.sqrt((((data-data.mean()) - (model-model.mean()))**2).mean())
    #print pattern_rms_diff
    
    print ('   > N= ', len(model))
    
    #nn = 1
    #ax  = axgrid[nn]
    #pr.plot_scatter(ax, data, model+bias, var=defs['elev'],color='b',nn=nn, title=base_info.cases[key]['label'])
    nn += 1
    #

plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
      wspace=0.7*wspace1, hspace=1.1 * hspace1)
plt.savefig(out_dir+ '/scatters_HWM' + ftype,dpi=dpi)
plt.close('all')

#sys.exit()
#
###################################################################

print ('Plot map for HWM data ...')
track = read_track(fname = base_info.track_fname)
##
var    = defs['elev']
vmin   = var['vmin']
vmax   = var['vmax']
dv     = (vmax-vmin)/50.0
levels = np.arange(vmin,vmax+dv,dv)
##

keys = np.sort(list(base_info.cases.keys()))
for key in keys:
    print ('  > ', key)
    maxelevf  =  base_info.cases[key]['dir'] + '/maxele.63.nc'
    ncmaxelev = n4.Dataset(maxelevf,'r')
    elev_max  = ncmaxelev.variables['zeta_max'][:]
    lon       = ncmaxelev.variables['x'][:]
    lat       = ncmaxelev.variables['y'][:]
    depth     = ncmaxelev.variables['depth'][:]
    zeta1     = ncmaxelev.variables['zeta_max'][:]
     
    mask      = zeta1 < 0
    zeta1[zeta1.mask] = 0.0
    val       = zeta1 
    ncmaxelev.close()

    #
    fig, ax = pr.make_map()
    fig.set_size_inches(9,9)
    
    lim = get_region_extent(region = base_info.regions[0])
    extent = [lim['xmin'],lim['xmax'],lim['ymin'],lim['ymax']+0]        
    ax.set_extent(extent)
    
    plot_track(ax,track,date=None)
    
    cond1 = ax.tricontour(tri,depth+0.1 ,levels=[0.0]  ,colors='k',lw=0.01, alpha= 0.5)
    
    if True:
        #cmap_   = cmaps.cmap_brightened (cmaps.jetMinWi,factor=0.75)
        cmap  =                       (cmaps.jetMinWi)
        #cmap  = cmaps.jetWoGn()
        cf1    = ax.tricontourf(tri,val,levels=levels, cmap = cmap , extend='both',alpha = 1.0)#extend='max' )  #,extend='both'
        #cf1   = ax.tripcolor(tri,val, cmap = cmap, vmin = vmin , vmax = vmax)#extend='max' )  #,extend='both'
        cb     = plt.colorbar(cf1,shrink = 0.15,ticks = [vmin,(vmin+vmax)/2,vmax])   
        cb.set_label('TWL [m]')
    
        # HWM points
        #ax.scatter(lon_hwm,lat_hwm,marker = 's',s=5,c=hwm,cmap=cmap_, zorder=3,edgecolors='None',alpha = 1.0)
        ax.scatter(xdata,ydata,marker = 's',s=15,c=data,cmap=cmap , zorder=3,alpha = 1.0)
    
    else:
        for ihwm in range(len(xdata)):
            ax.scatter(xdata[ihwm],ydata[ihwm],marker = 's',s=8,c='r',zorder=3,edgecolors='None',alpha = 0.75)
    
    #plot mesh
    if False:
        ax.triplot(tri, 'k-', lw=0.1, alpha=0.4)
    
        for ip in range(len(xmodel)):                                                                                    
            ax.plot([xmodel[ip],xdata[ip]],[ymodel[ip],ydata[ip]],'k-',lw=1)                                    
            #plt.plot(xdata.values[ip],ydata.values[ip],'ko')                                   
                                            
    titl = base_info.cases[key]['label'] 
    ax.set_title(titl)
    filename = out_dir + '/maps_hwm_obs_'+titl.replace(' ','_')+ftype
    plt.savefig(filename,dpi=450)
    plt.close('all')

#sys.exit()
    
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
    #key      = taylor_data.keys()[imodel]
    stddev   = taylor_data[key][0]
    corrcoef = taylor_data[key][1]
    marker   = ps.marker[imodel]
    dia.add_sample(stddev, corrcoef,ref=False, marker=marker, ls='', c=ps.colors[imodel],
               markersize=markersize,label=key)
     
# add refrence point for data     
dia.add_sample(refstd, 1.0 ,ref=True, marker='*', ls='',  c='k',
               markersize=markersize*1.5,label='Ref.')
 
# Add RMS contours, and label them
contours = dia.add_contours(levels=8,data_std=refstd,colors='0.5')
plt.clabel(contours, inline=1,fmt = '%.3g', fontsize=10)
 
# Add a figure legend
if True:
    leg2=fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='upper right',
               ncol=2)
 
    frame=leg2.get_frame()
    frame.set_edgecolor('None')
    #frame.set_facecolor('None')
    frame.set_facecolor('w')

plt.title( 'HWM data' + '  N=' + str(len(model)), position=(0.1, 1.04))
plt.subplots_adjust(left=left1, bottom=bottom1, right= right1, top= top1,
      wspace=wspace1, hspace=hspace1)
plt.savefig(out_dir+ '/taylor_HWM' + ftype,dpi=dpi)
plt.close('all')

####################################
print ('[info:] plot stats on fig ..')
params = ['cor', 'r2', 'rmse', 'rbias', 'bias', 'mae', 'peak', 'ia', 'skill']
params = ['cor','rmse', 'rbias', 'bias', 'peak', 'ia']
nall = len(params)

icol = 2
irow = nall//icol
params = params[:icol*irow]
#figure
fwidth,fheight= ps.get_figsize(300)
fwidth  = fwidth  *  icol *1.25
fheight = fheight *  irow 

fig,axgrid = plt.subplots(nrows=irow, ncols=icol, sharex=True, sharey=False,
  figsize=(fwidth, fheight),
  facecolor='w', edgecolor='k'
  #,gridspec_kw = {'width_ratios':[1, 1]} 
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
        #l = ax.plot(imodel,np.abs(stats_data[key][param]),marker=marker, ls='', c=ps.colors[imodel],
        #         markersize=markersize,label=key)
        l = ax.plot(imodel,(stats_data[key][param]),marker=marker, ls='', c=ps.colors[imodel],
                 markersize=markersize,label=key)
        
        #labs.append(key[9:-6])
        labs.append(key[:])
        
        samplePoints.append(l)
        imodel += 1
        
    #ylab = string.capitalize ('abs('+param+')' )  
    ylab = param.capitalize()  

    ax.set_ylabel( ylab)
    #plt.setp( ax, 'xticklabels', [] )
    ax.locator_params(axis='y', nbins=4)
    ax.xaxis.set_ticks(ticks=range(imodel)) 
    ax.xaxis.set_ticklabels(ticklabels=labs)  #,fontsize=18)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_rotation(90)
    
    ax.set_xlim(-0.2,imodel+0.2)
    ax.grid()

if False:
    leg2=ax.legend(numpoints=1, prop=dict(size='small'), loc='upper right',ncol=1)
    frame=leg2.get_frame()
    frame.set_edgecolor('None')
    frame.set_facecolor('None')
        
fig.suptitle('HWM data' + '  N=' + str(len(model)))
plt.subplots_adjust(left = 1.5 * left1, bottom = 2.5 * bottom1, right=right1, top=top1,
      wspace=4 * wspace1, hspace=hspace1)
plt.savefig(out_dir+ '/all_stat_HWM' + ftype,dpi=dpi)
plt.close('all')

#dfa.drop('lon',axis=1,inplace=True)
#dfa.drop('lat',axis=1,inplace=True)

pandas_plots = False

if pandas_plots:
   
    ###
    ext_min = defs['elev']['vmin']
    ext_max = defs['elev']['vmax']
    #############
    print ('[info:] plot matrix')
    plt.figure()
    g = sns.PairGrid(dfa, diag_sharey=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_upper(plt.scatter)
    g.map_diag(sns.kdeplot, lw=1)
    plt.savefig(out_dir + '/'+ '0pair_grid_HWM_'+str(ext_min)+ ftype,dpi=dpi)              
    plt.close('all')
    
    from pandas.tools.plotting import scatter_matrix
    scatter_matrix(dfa)
    plt.savefig(out_dir + '/'+ '0scatter_matrix_HWM_'+str(ext_min)+ ftype,dpi=dpi)              
    plt.close('all')    

    #pandas plot
    x1 = [-10,10]
    y1 = [-10,10]
    ####
    plt.close('all')
    fig = plt.figure(1,figsize=(6,7))
    sns.boxplot(data=dfa,whis=999999)    ## Incluse all numbers no outliers
    ax = plt.gca()
    #ax.set_ylim(ext_min*0.75,ext_max)
    ax.set_ylabel('Elev. [m]')
    plt.xticks(rotation =90)
    plt.subplots_adjust(left=left1, bottom=2.2 * bottom1, right=right1, top=top1,
                        wspace=1 * wspace1, hspace= 1 * hspace1)

    plt.savefig(out_dir + '/'+ '0box_plot_HWM_'+str(ext_min)+ ftype,dpi=dpi)              
    plt.close()
    #from pandas import scatter_matrix
    #scatter_matrix(dfa)

    ######
    plt.figure()
    #for y in dfa.columns[:]:
    for y in model_keys:
        if  y != 'data':
            reg = sns.regplot(x=y , y='data',data=dfa, x_estimator=np.mean,label=y)
    
    ax = plt.gca()
    ax.set_ylim(ext_min,ext_max)
    ax.set_xlim(ext_min,ext_max)
    ax.plot(x1,y1,'k',lw=0.5)
    ax.set_xlabel('models')
    ax.set_aspect(1)
    ax.legend()
    plt.savefig(out_dir + '/'+ '0reg_plot_HWM_'+str(ext_min)+  ftype,dpi=dpi)            
    plt.close()
    ######
    plt.figure()
    #for y in dfa.columns[1:]:
    for y in model_keys:
        j = sns.jointplot(x=y, y='data',data=dfa,kind='reg') 
        j.ax_joint.set_ylim(ext_min,ext_max)
        j.ax_joint.set_xlim(ext_min,ext_max)
        j.ax_joint.plot(x1,y1,'k',lw=0.5)
        j.ax_joint.text(ext_min+ 0.05 * (ext_max-ext_min), ext_min + 0.9 * (ext_max-ext_min), 'N='+str(len(dfa.data)))
        j.savefig(out_dir + '/'+ '0joint_plot_HWM_'+str(ext_min)+y+  ftype,dpi=dpi)    
        plt.close()      

    plt.figure()
    g = sns.pairplot(dfa,x_vars=model_keys,y_vars=['data'],kind='reg',size=5, aspect=1 )
    for ax in np.array(g.axes).squeeze():
        ax.set_ylim(ext_min,ext_max)
        ax.set_xlim(ext_min,ext_max)
        ax.plot(x1,y1,'k',lw=0.5)

    plt.savefig(out_dir + '/'+ '0pair_plot_HWM_'+str(ext_min)+  ftype,dpi=dpi)          
    plt.close('all')      
     

    dfa_diff = dfa * 1.0

    for col in model_keys:
        dfa_diff[col] =  dfa[col] - dfa['data']


    t = dfa_diff.describe()

#####




latp = 29.2038
lonp = -92.2285
#613979


#latp = 29.2
#lonp = -94.2
#570177


i,prox = find_nearest1d(xvec = lon,yvec = lat,xp = lonp,yp = latp)



