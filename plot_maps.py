from __future__ import division
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot multi axes ####
Dec 2021: adapter for Py3.x

"""
__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2017, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"

#Wed 24 May 2017 03:12:39 PM UTC 

#import matplotlib as mpl
#mpl.use('Agg')
import os,sys
#sys.path.append('/home/moghimis/linux_working/00-working/04-test-adc_plot/')
#sys.path.append('/home/moghimis/linux_working/00-working/04-test-adc_plot/csdlpy')


import matplotlib
if os.name == 'nt':
    matplotlib.rc('font', family='Arial')
else:  # might need tweaking, must support black triangle for N arrow
    matplotlib.rc('font', family='DejaVu Sans')

import netCDF4
import numpy as np
import matplotlib.pyplot as plt

import datetime
#from   collections import defaultdict
from   pynmd.plotting.vars_param import defs
from   pynmd.models.adcirc.post import adcirc_post as adcp
from   pynmd.plotting import plot_routines as pr
from   pynmd.plotting import plot_settings as ps
from   pynmd.tools.gtime import roundTime
from   pynmd.tools.gtime import find_nearest_time
from   pynmd.plotting import colormaps as cmaps
import pandas as pd

import glob
from dateutil import parser

#sys.path.append('/scratch2/COASTAL/coastal/save/Saeed.Moghimi/opt/pycodes/csdlpy')
#import adcirc
from csdlpy.atcf import readTrack

try:
    os.system('rm base_info.pyc'  )
except:
    pass
if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
import base_info

defs = base_info.defs.copy()


try:
    os.system('rm geo_regions.pyc'  )
except:
    pass
if 'geo_regions' in sys.modules:  
    del(sys.modules["geo_regions"])
from geo_regions import get_region_extent    

#import matplotlib.tri as Tri
#from dateutil import parser
#import glob

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER,
                                   LATITUDE_FORMATTER)






#====== subplot adjustments ===============
left1  = 0.1    # the left side of the subplots of the figure
right1 = 0.8   # the right side of the subplots of the figure
bottom1= 0.1    # the bottom of the subplots of the figure   (ntr==16   bottom=0.05)
top1   = 0.9    # the top of the subplots of the figure
wspace1= 0.05   # the amount of width reserved for white space between subplots
hspace1= 0.15   # the amount of height reserved for white space between subplots
###############################################<<<<<<


try:
   ftypes = base_info.ftypes
except:
   ftypes = ['png','pdf']
   ftypes = ['png']    

def plot_map_old(ax,tri,val,extent,vmin=None,vmax=None,dep=None,cmap=None):

    try:
        #if cartopy
        ax.set_extent(extent)
    except:
        ax.set_xlim(extent[0],extent[1]) 
        ax.set_ylim(extent[2],extent[3]) 
        
    if cmap is not None:
        cmap = cmap
    else:
        cmap = cmaps.jetMinWi
    
    if vmin is not None:
        #print ' tricontourf '
        dv = (vmax-vmin)/50.0
        levels=np.arange(vmin,vmax+dv,dv)
        #val [val < vmin] = vmin+1e-3
        cf1 = ax.tricontourf(tri,val,levels=levels, cmap = cmap , extend='both')#extend='max' )  #,extend='both'
        plt.colorbar(cf1,shrink = 0.4,ticks = [vmin,(vmin+vmax)/2,vmax])      
    else:
        cf1 = ax.tricontourf(tri,val , cmap = cmap)
        plt.colorbar(cf1,shrink = 0.4)      


    if dep is not None:
        cond1 = ax.tricontour(tri,dep+0.1 ,levels=[0.0]  ,colors='k',lw=0.5)
        cond2 = ax.tricontour(tri,dep+0.1 ,levels=[50.0],colors='k',lw=1, linestyles='dashed')
     
    if True:
        con1 = ax.tricontour(tri,val ,levels=[0.25 * val.max()],colors='k',alpha=1,lw=2)
        con2 = ax.tricontour(tri,val ,levels=[0.5]            ,colors='r',alpha=1,lw=2)
        con3 = ax.tricontour(tri,val ,levels=[0.25]           ,colors='g',alpha=1,lw=2)
         
        #plt.clabel(con1, con1.levels,inline=True,fmt='%1.1g',fontsize=6)
        #plt.clabel(con2, con2.levels,inline=True,fmt='%1.1g',fontsize=6)
        #plt.clabel(con2, con2.levels,inline=True,fmt='%1.1g',fontsize=6)

        xtext = extent[0] + 0.6 * (extent[1] - extent[0])
        ytext = extent[2] + 0.05 * (extent[3] - extent[2])
    
        t = ax.text(xtext, ytext , r'0.25 * Max(Surge) [m] ', color = 'k',fontsize=7)
        t.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='w'))
        

        xtext = extent[0] + 0.75 * (extent[1] - extent[0])
        ytext = extent[2] + 0.15 * (extent[3] - extent[2])

        t = ax.text(xtext, ytext , r'0.5 [m] '               , color = 'r',fontsize=7)
        t.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='w'))


        xtext = extent[0] + 0.6  * (extent[1] - extent[0])
        ytext = extent[2] + 0.15 * (extent[3] - extent[2])

        t = ax.text(xtext, ytext , r'0.25 [m] '              , color = 'g',fontsize=7)
        t.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='w'))






def plot_map(ax=None,tri=None,val=None,var=None,lim=None,dep=None, pres=None,no_axes_label=True,plot_mesh=True, scale_bar = True, u=None,v=None):

    if lim is not None:
        extent = [lim['xmin'],lim['xmax'],lim['ymin'],lim['ymax']]        
        try:
            #if cartopy
            ax.set_extent(extent)
        except:
            ax.set_xlim(extent[0],extent[1]) 
            ax.set_ylim(extent[2],extent[3]) 
        
    if var is not None:
        cmap = var['cmap']
    else:
        cmap = cmaps.jetMinWi
    
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
    
    cb_dx = 0.9
    cb_dy = 0.3
    cb_size = 0.3
    cbax    = fig.add_axes([xsub1+ cb_dx * (xsub2-xsub1), ysub1 + cb_dy * (ysub2-ysub1), 0.01, cb_size]) 
    
    vmin = var['vmin']
    vmax = var['vmax']
    step = 0.5  #m
    levels = np.arange(vmin, vmax+step, step=step)
    #contour = ax.tricontourf(tri, zeta0,levels=levels,cmap = my_cmap ,extend='max')
    cf1 = ax.tricontourf(tri,zeta0,levels=levels, cmap = cmap , extend='both')#extend='max' )  #,extend='both'  
    cb  = plt.colorbar(cf1,cax=cbax,ticks = [vmin,(vmin+vmax)/2,vmax],format='%1.4g',orientation='vertical')      
    cb.set_label(var['label'])

    if base_info.vec:
        vecp = ax.quiver(tri.x, tri.y, u, v,
          units='xy', scale=10., zorder=3, color='k',
          width=0.007, headwidth=3., headlength=4.)

    if dep is not None:
        cond1 = ax.tricontour(tri,dep+0.1 ,levels=[0.0]  ,colors='k',lw=0.02,alpha=0.8)
        #cond2 = ax.tricontour(tri,dep+0.1 ,levels=[25.0]  ,colors='k',lw=0.2, linestyles='dashed')
        #cond3 = ax.tricontour(tri,dep+0.1 ,levels=[50.0] ,colors='k',lw=0.2, linestyles='dashed')

    if base_info.plot_mesh:
        ax.triplot(tri, 'k-', lw=0.1, alpha=0.6)
    
    if pres is not None:
        indp = np.argmin(pres)
        #ax.plot(tri.x[indp],tri.y[indp],'ro',alpha=1,ms=8)
    
    #location of Burbuda
    #ax.plot(17.57,-61.80,'ko',alpha=1,ms=8,zorder=3)
        #cond1 = ax.tricontour(tri,pres ,levels=[pres.min() * 1.1]  ,colors='k',lw=1)

    try:
        if val.sum() !=0:
            ind_max = np.array(np.where((val == val.max()))).squeeze().item()
            #ax.plot(tri.x[ind_max],tri.y[ind_max],'m*',ms=14)
        else:
            print ('val is zero !!!')
    except:
        pass
         
    if False:
        con1 = ax.tricontour(tri,val ,levels=[0.25 * val.max()],colors='k',alpha=1,lw=2)
        con2 = ax.tricontour(tri,val ,levels=[0.5]             ,colors='r',alpha=1,lw=2)
        con3 = ax.tricontour(tri,val ,levels=[0.25]            ,colors='g',alpha=1,lw=2)
         
        #plt.clabel(con1, con1.levels,inline=True,fmt='%1.1g',fontsize=6)
        #plt.clabel(con2, con2.levels,inline=True,fmt='%1.1g',fontsize=6)
        #plt.clabel(con2, con2.levels,inline=True,fmt='%1.1g',fontsize=6)

        xtext = extent[0] + 0.6 * (extent[1] - extent[0])
        ytext = extent[2] + 0.05 * (extent[3] - extent[2])
    
        t = ax.text(xtext, ytext , r'0.25 * Max(Surge) [m] ', color = 'k',fontsize=7)
        t.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='w'))
        

        xtext = extent[0] + 0.75 * (extent[1] - extent[0])
        ytext = extent[2] + 0.15 * (extent[3] - extent[2])

        t = ax.text(xtext, ytext , r'0.5 [m] '               , color = 'r',fontsize=7)
        t.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='w'))


        xtext = extent[0] + 0.6  * (extent[1] - extent[0])
        ytext = extent[2] + 0.15 * (extent[3] - extent[2])

        t = ax.text(xtext, ytext , r'0.25 [m] '              , color = 'g',fontsize=7)
        t.set_bbox(dict(facecolor='w', alpha=0.75, edgecolor='w'))
        
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=30 )

    if scale_bar:
        pr.scale_bar(ax, ax.projection, 10) 

    if no_axes_label:
        plt.setp( ax, 'xticklabels', [] )
        ax.set_xlabel('')
        plt.setp( ax, 'yticklabels', [] )
        ax.set_ylabel('') 
    
def datetime64todatetime(dt):
    tmp=[]
    for it in range(len(dt)):
        tmp.append(pd.Timestamp(dt[it]).to_pydatetime())
    return np.array(tmp)



def plot_track(ax,track,date=None,color = 'r'):
    """
    plot storm track
    """    
    ax.plot(track['lon'],track['lat'],lw=3,color=color,marker = 'None',ls='dashed',alpha=1)

    if date is not None:
        dates = np.array(track['dates'])
        #ind = np.array(np.where((dates==date))).squeeze().item()
        ind = find_nearest_time(dates,date)
        ax.plot(track['lon'][ind],track['lat'][ind],marker ='o',color='k', alpha=0.6,ms=8,)
    
def get_diff(varname,ind0,ind1):
    fname = base_info.cases[base_info.key0]['dir'] + defs[varname]['fname']
    nc0   = netCDF4.Dataset(fname)
    ncv0  = nc0.variables
    val0  = ncv0[defs[varname]['var']][ind0]
    #
    fname = base_info.cases[base_info.key1]['dir'] + defs[varname]['fname']
    nc1   = netCDF4.Dataset(fname)
    ncv1  = nc1.variables
    val1  = ncv1[defs[varname]['var']][ind1]

    return (val1 - val0)
    
def get_val(varname,ind1=None):
    #
    
    fname = base_info.cases[base_info.key1]['dir'] + defs[varname]['fname']
    nc1   = netCDF4.Dataset(fname)
    ncv1  = nc1.variables
    if ind1 is None:
        val1  = ncv1[defs[varname]['var']][:]
    else:
        val1  = ncv1[defs[varname]['var']][ind1]
    
    nc1.close()
    return val1

def get_uv(ind1=None):
    #
    
    fname = base_info.cases[base_info.key1]['dir'] + 'fort.64.nc'
    nc1   = netCDF4.Dataset(fname)
    ncv1  = nc1.variables
    if ind1 is None:
        u  = ncv1['u-vel'][:]
        v  = ncv1['v-vel'][:]

    else:
        u  = ncv1['u-vel'][ind1]
        v  = ncv1['v-vel'][ind1]
    
    nc1.close()
    return u,v



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
    
    trc = trc.drop_duplicates(subset='dates')#,keep='first')
    trc = trc.set_index (trc.dates)
    trc = trc.resample('H').interpolate()
    trc.drop('dates',axis=1,inplace=True)
    
    dates = datetime64todatetime(trc.index)
    
    return dict(dates=dates,lon=trc.lon.values, lat=trc.lat.values)
    
#################################################################################################
if False:
    if base_info.storm_year == 'IKE':
        #read IKE track
        track = read_track()
    
        include_gustav = False
        #read Gustav track
        if include_gustav:
            gustav_fname = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/data/tracks/gustav_aal072008.dat'
            gus_track = read_track(fname=gustav_fname)
    
    if base_info.storm_year == 'ISA':
        print (' oopppss ..') 

    isabel_fname = '/scratch4/COASTAL/coastal/save/Saeed.Moghimi/models/NEMS/NEMS_inps/01_data/tracks/isabel_aal132003.dat'
    track = read_track(fname=isabel_fname)

## back up scripts
out_dir = base_info.out_dir + '/maps/'
scr_dir = out_dir + '/scr/'
os.system('mkdir -p ' + scr_dir)
args=sys.argv
scr_name = args[0]
os.system('cp -fr  '+scr_name +'    '+scr_dir)
os.system('cp -fr     *.py     '     + scr_dir)
#
# read base_run only_tide time info
fname  = base_info.cases[base_info.key0]['dir'] + '/fort.63.nc'
nc0    = netCDF4.Dataset(fname)
ncv0   = nc0.variables
depth  = ncv0['depth'][:]
depth [depth < -10] = -10
dates0 = netCDF4.num2date(ncv0['time'][:],ncv0['time'].units) 
indd0   = np.array(np.where((dates0 > base_info.tim_lim['xmin'] )&(dates0<base_info.tim_lim['xmax'])  )).squeeze()
nc0.close()
#
fname  = base_info.cases[base_info.key1]['dir'] + '/fort.63.nc'
nc1     = netCDF4.Dataset(fname)
ncv1    = nc1.variables
dates1 = netCDF4.num2date(ncv1['time'][:],ncv1['time'].units) 
indd1   = np.array(np.where((dates1 > base_info.tim_lim['xmin'] )&(dates1<base_info.tim_lim['xmax'])  )).squeeze()
nc1.close()
#
#construct HSOFS Tri mask
lon,lat,tri  = adcp.ReadTri(base_info.cases[base_info.key0]['dir'])
#                    
print (base_info.cases[base_info.key1]['dir'])
#
track = read_track(fname=base_info.track_fname)

if base_info.plot_nwm_files:
    #read channel points locations
    geo_info      = netCDF4.Dataset(base_info.nwm_channel_pnts)
    geo_ncv = geo_info.variables
    lat_nwm = geo_ncv['latitude'][:]
    lon_nwm = geo_ncv['longitude'][:]

    # read channel connections
    geo_poly_nwm  = pd.read_csv(base_info.nwm_channel_geom)

#for region in ['hsofs_region','ike_region','ike_local']:
#for region in ['ike_region']:
#for varname in ['elev']:   
if base_info.plot_adc_fort:
    for varname in base_info.varnames:   
        for ind1 in indd1[::-1][:]:
            print ('-------------------------------------------')
            for region in base_info.regions:
                lim = get_region_extent(region = region)
                date  = dates1[ind1]
                # find matching index for new netcdf file 

                print (ind1, region ,varname, date)
                
                try:
                    pres =  get_val('pmsl',ind1)
                except:
                    pres = None
                    pass   
                
                title1 = base_info.cases[base_info.key1]['label']
                if varname in ['elev','u','v']:
                    if base_info.dif:
                        ind0  = np.array(np.where((dates0== date))).squeeze().item()
                        val  = get_diff(varname,ind0,ind1)
                        title1 = base_info.cases[base_info.key1]['label']+' - ' +\
                        base_info.cases[base_info.key0]['label']
                    else:
                         val = get_val(varname,ind1)
                
                
                    if base_info.vec:
                        u,v = get_uv(ind1=None)
                
                if varname in ['rad']:
                    radx = 'radstress_x'
                    rady = 'radstress_y'
                    #
                    defs[varname]['var'] = radx
                    valx = get_val(varname,ind1)
                    
                    defs[varname]['var'] = rady
                    valy = get_val(varname,ind1)

                    val = np.sqrt(valx*valx+valy*valy)

                                   
                if varname in ['wind']:
                    radx = 'windx'
                    rady = 'windy'
                    #
                    defs[varname]['var'] = radx
                    valx = get_val(varname,ind1)
                    
                    defs[varname]['var'] = rady
                    valy = get_val(varname,ind1)
                    
                    val = np.sqrt(valx*valx+valy*valy)

                #if 'max' in varname:
                #    defs['maxelev'][]
                
                val[np.isnan(val)] = 0.0
                fig, ax = pr.make_map()
                #fig = plt.figure()
                dx = abs(lim['xmin'] - lim['xmax'])
                dy = abs(lim['ymin'] - lim['ymax'])
                
                fig.set_size_inches(9,9*1.4*dy/dx)
                #plot_map (ax,tri,val,extent,vmin=vmin,vmax=vmax, dep=dep , cmap = cmap)
                #sys.exit()
                
                if not base_info.vec:
                    plot_map (ax=ax,tri=tri,val=val,var=defs[varname],lim=lim,dep=depth, pres=pres,plot_mesh=base_info.plot_mesh)
                else:
                    plot_map (ax=ax,tri=tri,val=val,var=defs[varname],lim=lim,dep=depth, pres=pres,plot_mesh=base_info.plot_mesh,u=u,v=v)

                #track = readTrack(base_info.track_fname)
                #ax.plot(track['lon'],track['lat'],'r-',lw=2,alpha=1,ms=10, markevery = 5, marker = r'$\bigotimes$',mec='None',mew=2)  #marker = r'$\S$'

                plot_track(ax,track,date=date, color= 'r')
                
                #find val max in the bounding box
                [ind_box] = np.where((lon > defs['lim']['xmin']) & 
                    (lon < defs['lim']['xmax']) & 
                    (lat > defs['lim']['ymin']) & 
                    (lat < defs['lim']['ymax']))
                ind_max = np.argmax(val[ind_box])
                lon_max = lon[ind_box][ind_max]
                lat_max = lat[ind_box][ind_max]
                val_max = val[ind_box][ind_max]
                ax.scatter(lon_max,lat_max,c='m',s=15,zorder=3)
                
                #ax.set_title(base_info.storm_name + ' ' +base_info.storm_year+' '+base_info.cases[base_info.key1]['label']+' - ' +\
                #    base_info.cases[base_info.key0]['label']+'\n  '+defs[varname]['label'] +\
                #    ' Date: ' + date.isoformat() +'\n Max. Val. =  %.2g' %val.max() + '[m]')
                
                ax.set_title(base_info.name + ' ' +base_info.year+' '+title1+'\n'+ ' Date: ' + date.isoformat() +'\n'+' Max. Val. =  %.2g' %val_max)
                
                
                date_str = date.isoformat().replace(':','-')
                
                del val
                out_dir1  = out_dir + '/' + region + '_' + varname
                os.system('mkdir -p '+ out_dir1)
                
                plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
                            wspace=wspace1, hspace=hspace1)

                filename = out_dir1 + '/adc_fort_files_'+region+ '_' +varname.capitalize()+str(10000+ind1)+'_'+date_str+'.png'
                plt.savefig(filename,dpi=150)
                #plt.show()
                plt.close('all')
            
            os.system('cp -fr  '+scr_name +'    '+out_dir1)
            os.system('cp -fr  base_info*.py    '+out_dir1)
        #else:

if base_info.plot_adc_maxele:
    varname = 'elev'
    for region in base_info.regions:
        print ('Maxele diff ', region) 
    
        lim = get_region_extent(region = region)
        print (lim) 
        fname  =  base_info.cases[base_info.key0]['dir'] + '/maxele.63.nc'
        nc0    = netCDF4.Dataset(fname)
        ncv0   = nc0.variables
        zeta0  = ncv0['zeta_max'][:]
        dep0   = ncv0['depth'][:]
        lon0   = ncv0['x'][:]
        lat0   = ncv0['y'][:]
        elems  = ncv0['element'][:,:]-1  # Move to 0-indexing by subtracting 1
        zeta0[zeta0.mask] = 0.0
        #tri = Tri.Triangulation(lon0,lat0, triangles=elems)
        
        
        date = None
        ##
        fname  =  base_info.cases[base_info.key1]['dir'] + '/maxele.63.nc'
        nc1     = netCDF4.Dataset(fname)
        ncv1    = nc1.variables
        zeta1  = ncv1['zeta_max'][:]
        
        mask = zeta1 < 0


       
        val = zeta1 - zeta0
        val = np.ma.masked_where(val<0,val)
        val[val==mask] = 0.0
        val[np.isnan(val)] = 0.0
        
        fig, ax = pr.make_map()

        if False:    
            vmin    = base_info.defs['elev']['vmin']
            vmax    = base_info.defs['elev']['vmax'] 
            step = 0.5  #m
            levels = np.arange(vmin, vmax+step, step=step)
            #contour = ax.tricontourf(tri, zeta0,levels=levels,cmap = my_cmap ,extend='max')
            cf1 = ax.tricontourf(tri,zeta0,levels=levels, cmap = my_cmap , extend='both')#extend='max' )  #,extend='both'
            
            # set scale for vertical vector plots
            pos_ax   = np.array (ax.get_position ())
            aheight  = pos_ax[1][1] -pos_ax[0][1]
            awidth   = pos_ax[1][0] -pos_ax[0][0]

            fwidth   = 1
            fheight  = 1
            wscale  = aheight*fheight/(awidth*fwidth) * (lim['xmax']-lim['xmin'])/(lim['ymax']-lim['ymin'])
            #
            xsub1 = pos_ax[0][0]
            xsub2 = pos_ax[1][0]
            ysub1 = pos_ax[0][1]            
            ysub2 = pos_ax[1][1]

            cb_dx = 0.9
            cb_dy = 0.3
            cb_size = 0.3
            cbax    = fig.add_axes([xsub1+ cb_dx * (xsub2-xsub1), ysub1 + cb_dy * (ysub2-ysub1), 0.01, cb_size]) 
            cb  = plt.colorbar(cf1,cax=cbax,ticks = [vmin,(vmin+vmax)/2,vmax],format='%1.4g',orientation='vertical')    

        dx = abs(lim['xmin'] - lim['xmax'])
        dy = abs(lim['ymin'] - lim['ymax'])
        
        fig.set_size_inches(9,9*1.4*dy/dx)

        #fig.set_size_inches(9,9)
        plot_map (ax=ax,tri=tri,val=val,var=defs[varname],lim=lim,dep=depth, pres=None,plot_mesh=base_info.plot_mesh)
        #sys.exit()
        plot_track(ax,track,date=None, color= 'r')

        #find val max in the bounding box
        [ind_box] = np.where((lon > lim['xmin'])  &
                             (lon < lim['xmax']) &
                             (lat > lim['ymin']) &
                             (lat < lim['ymax']))
        ind_max = np.argmax(val[ind_box])
        lon_max = lon[ind_box] [ind_max]
        lat_max = lat[ind_box] [ind_max]
        val_max = val[ind_box] [ind_max]
        if False:
            #plot maximum surge location
            ax.scatter(lon_max,lat_max,c='m',s=15,zorder=3)
                

        
        out_dir1  = out_dir + '/' + region + '_maxelev_' 
        os.system('mkdir -p '+ out_dir1)

        ax.set_title(base_info.name + ' ' +base_info.year+' '+base_info.cases[base_info.key1]['label']+' - ' +\
                     base_info.cases[base_info.key0]['label'])#+'\n  '+defs[varname]['label']   \
                     #+'\n '+' Max. Val. =  %.2g' %val.max() + '[m]'  )
        

        
        if base_info.plot_nwm_files:
            #need to be based on date or/XX
            chan       = netCDF4.Dataset(base_info.nwm_results_dir + '/short_range/' + 'nwm.t02z.short_range.channel_rt.f018.conus.nc')
            chan_ncv   = chan.variables
            f_ids      = chan_ncv['feature_id'][:]
            streamflow = chan_ncv['streamflow'][:]    
       
            df  = geo_poly_nwm[(geo_poly_nwm['xlat_mid'] > lim['xmin']) & (geo_poly_nwm['xlat_mid'] < lim['xmax']) & (geo_poly_nwm['ylon_mid'] > lim['ymin'])& (geo_poly_nwm['ylon_mid'] < lim['ymax'])]
            mask = (lon_nwm > lim['xmin']) & (lon_nwm < lim['xmax']) & (lat_nwm > lim['ymin'])& (lat_nwm < lim['ymax'])
            lon_df = lon_nwm[mask]
            lat_df = lat_nwm[mask]
            #ax.plot(lon_df,lat_df,'o',color = 'k',ms=0.5, lw=0.5,label= str(df.loc[fid]['feature_id']))
            
            all_ind = [] 
            for fid in df['FID']:
                [ind] = np.where(f_ids==df.loc[fid]['feature_id'])
                all_ind.append(ind.item())

            df['streamflow'] = streamflow[all_ind]
            df_sorted = df.sort('streamflow')
            df_sorted [df_sorted['streamflow']<10] = np.nan
            df_sorted =  df_sorted.dropna()  
            
            lws = np.linspace(0.25,4,len(df_sorted))
            
            il = 0
            for fid in df_sorted['FID']:
                lontmp = [df_sorted.loc[fid]['xlat_up'],df_sorted.loc[fid]['xlat_mid'],df_sorted.loc[fid]['xlat']]
                lattmp = [df_sorted.loc[fid]['ylon_up'],df_sorted.loc[fid]['ylon_mid'],df_sorted.loc[fid]['ylon']]
                ax.plot(lontmp,lattmp,color='b', lw = lws [il] ,label= str(df_sorted.loc[fid]['feature_id']),alpha = 0.6)
                il += 1
                
                #q_lateral  = chan_ncv['q_lateral'][:]
                #velocity   = chan_ncv['velocity'][:]
                
        plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
                            wspace=wspace1, hspace=hspace1)
        for ftype in ftypes:
            filename = out_dir1 + '/maxele'+region+ '_maxelev' +'.' + ftype 
            plt.savefig(filename,dpi=600)
        #plt.show()
        #sys.exit()
        plt.close('all')


if base_info.plot_nems_fields:
    #varname = 'sxx'
    for varname in base_info.varnames:
        if varname == 'pmsl':
            file_piece = '/field_ocn_pmsl*'
        
        if varname == 'sxx':
            file_piece = '/field_ocn_sxx*'
            defs[varname]          = {}
            defs[varname]['cmap']  = cmaps.jetMinWi
            defs[varname]['label'] = 'Sxx'
        
        plist = np.sort(glob.glob( base_info.cases[base_info.key1]['dir']  + file_piece))
        
        for ind1 in range(0,len(plist)-1)[::-1]:
            for region in base_info.regions:
            #for region in  ['hsofs_region']:
                
                lim = get_region_extent(region = region)
                fname = plist[ind1]
                date = parser.parse(fname[-22:-3])
                print (date)
                if ((date> base_info.tim_lim['xmin'] ) &(date<base_info.tim_lim['xmax'])):
                    print (region, fname) 
                    #
                    nc   = netCDF4.Dataset(fname) 
                    val  = nc.variables[varname][:]
                    #
                    ### NOUPC pressure
                    if varname == 'sxx':
                        defs[varname]['vmin']  =  0
                        defs[varname]['vmax']  =  5000
                        val[val > 1e10] = 0.0
                
                    if varname == 'pmsl':
                        #defs[varname]['vmin']  =  92000
                        #defs[varname]['vmax']  =  99000
                        defs[varname]['vmin']  =  92000
                        defs[varname]['vmax']  =  105000
                    
                    ### plot stuff
                    fig, ax = adcp.make_map()
                    dx = abs(lim['xmin'] - lim['xmax'])
                    dy = abs(lim['ymin'] - lim['ymax'])
                    
                    fig.set_size_inches(9,9*1.4*dy/dx)
                    
                    plot_map (ax=ax,tri=tri,val=val,var=defs[varname],lim=lim,dep=depth)
                    plot_track(ax,track,date=date)
                    
                    title1 = base_info.storm_year + \
                             '\n'+varname +' Date: ' + date.isoformat() +'\n Max. Val. =  %.2g' %val.max()
                             #base_info.cases[base_info.key1]['label']+',  '+defs[varname]['label'] +\

                    ax.set_title(title1)
                    date_str = date.isoformat().replace(':','-')
                    
                    del val
                    out_dir1  = out_dir + '/nems_fields_' + region + '_' + varname
                    os.system('mkdir -p '+ out_dir1)
                   
                    plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
                                        wspace=wspace1, hspace=hspace1)

                
                    filename = out_dir1 + '/'+region+ '_' +varname.capitalize()+str(10000+ind1)+'_'+date_str+'.png'
                    plt.savefig(filename,dpi=450)
                    #plt.show()
                    plt.close('all')
                    nc.close()
                
                os.system('cp -fr  '+scr_name +'    '+out_dir1)
                os.system('cp -fr  base_iny     '+out_dir1)



if base_info.plot_forcing_files:
    #varname = 'pmsl'
    #varname = 'hs'
    for varname in base_info.varnames:
        if varname in ['pmsl','wind']:
            print ('Plot HWRF input file ..')
            file_piece = '/inp_atmesh/*.nc'
            plist = np.sort(glob.glob( base_info.cases[base_info.key1]['dir']  + file_piece))
            fname = plist[0]
            nch    = netCDF4.Dataset(fname) 
            nchv   = nch.variables
            uwnd   = nchv['uwnd']
            vwnd   = nchv['vwnd']
            pres   = nchv['P']
            dates  = netCDF4.num2date(nchv['time'][:],nchv['time'].units) 
        
        if varname == 'hs':
            print ('Plot Hs WW3 Res file ..'  )            
            fname = base_info.cases[base_info.key1]['hsig_file']
            nch    = netCDF4.Dataset(fname) 
            nchv   = nch.variables
            hs     = nchv['hs']
            dates  = netCDF4.num2date(nchv['time'][:],nchv['time'].units) 
        
        
        for ind1 in range(0,len(dates)-1)[::-1]:
            for region in base_info.regions:
                lim = get_region_extent(region = region)
                date = dates[ind1]
                date = roundTime(dt=date, dateDelta=datetime.timedelta(minutes=1))
                print (date)
                if ((date> base_info.tim_lim['xmin'] ) &(date<base_info.tim_lim['xmax'])):
                    print (region, fname) 
                    #
                    #
                    ### NOUPC pressure
                    if varname == 'sxx':
                        defs[varname]['vmin']  =  0
                        defs[varname]['vmax']  =  5000
                        val[val > 1e10] = 0.0
                
                    if varname == 'pmsl':
                        val = pres[ind1]

                    
                    if varname == 'hs':
                        val = hs[ind1]            
                        #defs[varname]['vmin']  =  0
                        #defs[varname]['vmax']  =  16                    ### plot stuff
                        #defs[varname]['vmax']  =  val.max() // 5 * 5                    ### plot stuff


                    if varname in ['wind']:
                        valx = uwnd[ind1]
                        valy = vwnd[ind1]
                        val = np.sqrt(valx*valx+valy*valy)

                    fig, ax = adcp.make_map()
                    dx = abs(lim['xmin'] - lim['xmax'])
                    dy = abs(lim['ymin'] - lim['ymax'])
                    
                    fig.set_size_inches(9,9*1.4*dy/dx)
                    
                    plot_map (ax=ax,tri=tri,val=val,var=defs[varname],lim=lim,dep=depth,plot_mesh=base_info.plot_mesh)
                    plot_track(ax,track,date=date,color = 'r')
                    ax.set_title(base_info.storm_name + ' ' +base_info.storm_year +'  '+ base_info.cases[base_info.key1]['label']+\
                        '\n'+varname +' Date: ' + date.isoformat() +'\n Max. Val. =  %4.4g' %val.max() )
                    date_str = date.isoformat().replace(':','-')

                    plt.subplots_adjust(left=left1, bottom=bottom1, right=right1, top=top1,
                                        wspace=wspace1, hspace=hspace1)
                    
                    del val
                    out_dir1  = out_dir + '/forcings_' + region + '_' + varname
                    os.system('mkdir -p '+ out_dir1)
                
                    filename = out_dir1 + '/'+region+ '_' +varname.capitalize()+str(10000+ind1)+'_'+date_str+'.png'
                    plt.savefig(filename,dpi=300)
                    #plt.show()
                    plt.close('all')
        
        
        nch.close()
        os.system('cp -fr  '+scr_name +'    '+out_dir1)
        os.system('cp -fr          *.py     '+out_dir1)            
            
    if False:     
        indd = range(0,len(dates)-1)[::-1]
        for ind1 in indd[::-1]:
            date =dates[ind1] 
            date_str = date.isoformat()
            print (date_str)
            fig, ax = adcp.make_map()
            fig.set_size_inches(7,7)
            extent = [lim['xmin'],lim['xmax'],lim['ymin'],lim['ymax']]        
            ax.set_extent(extent)
        
            ind_time = find_nearest_time(dates,date)
            color = 'red'
            #ax.plot(track['lon'][ind_time],track['lat'][ind_time],'ro',alpha=1,ms=8)
            #ax.plot(track['lon'],track['lat'],lw=3,color=color,marker = '*', ms = 2, ls='dashed',alpha=1)
        
        
            plot_track(ax,track,date=date)
            out_dir1  = out_dir + '/atmesh_only_track_'+'_' + region 
            os.system('mkdir -p '+ out_dir1)
            
            filename = out_dir1 + '/'+region+ '_' +varname.capitalize()+str(10000+ind1)+'_'+date_str+'.png'
            plt.savefig(filename,dpi=450)
            plt.close('all')
            
    
    


















"""  


varname = 'rad'

fname = base_info.cases[base_info.key1]['dir'] + defs[varname]['fname']
nc1   = netCDF4.Dataset(fname)
ncv1  = nc1.variables

radx_2d = ncv1[defs[varname]['var']+'_y'][214]        

vmax,vmin=1e-4,-1e-4
dv = (vmax-vmin)/50.0
levels=np.arange(vmin,vmax+dv,dv)
tricontourf(tri,radx_2d,levels=levels,cmap=maps.jetWoGn())





latp = 29.2038
lonp = -92.2285
#613979


radxx   = ncv1[defs[varname]['var']+'_y'][:,613979]
plt.figure()

plt.plot(radxx[2:])

radxx   = ncv1[defs[varname]['var']+'_y'][:,613978]


plt.plot(radxx[2:])



    fname  = cases[key0]['dir'] + '/maxele.63.nc'
    nc0    = netCDF4.Dataset(fname)
    ncv0   = nc0.variables
    zeta0  = ncv0['zeta_max'][:]
    dep0   = ncv0['depth'][:]
    date = None
    ##
    fname  = cases[key1]['dir'] + '/maxele.63.nc'
    nc1     = netCDF4.Dataset(fname)
    ncv1    = nc1.variables
    zeta1  = ncv1['zeta_max'][:]
    
    mask = zeta1 < 0
    
    #tri.set_mask = maskDryElementsTri(tri,mask)
   
    val = zeta1 - zeta0
    #val = np.ma.masked_where(val<0,val)
    
    if False:
        fig, ax = adcp.make_map()
        fig.set_size_inches(7,7)
    else:
        fig = plt.figure()
        fig.set_size_inches(7,7)
        ax = plt.gca()
        
        #lon0 = -101.25
        #lon1 = -74.53125
        #lat0 = 13.9234038977
        #lat1 = 33.1375511923
        #fpng = 'lon0_lat0_lon1_lat1_-101.25_-74.53125_13.9234038977_33.1375511923.png'
    
        lon0 = -180.0
        lon1 =  180.0
        lat0 = -90.0
        lat1 =  90.0
        fpng = 'BMNG_hirez.png'
        #fpng = 'world.200410.3x21600x10800.jpg'
        
        img = plt.imread(fpng)
        ax.imshow(img,extent=[lon0,lon1,lat0,lat1],origin='upper')

    plot_map (ax,tri,val,extent,vmin=vmin,vmax=vmax, dep=dep , cmap = cmap)
    plot_track(ax,ike_track)
    
    if include_gustav:
        plot_track(ax,gus_track,color='b')


    ax.set_title(cases[key1]['label'] + '\n' + 'Max. Val. =  %.2g' %val.max() + '[m]')
    filename = out_dir1 + '/maxelev_'+region+ '_' +'.png'
    plt.savefig(filename,dpi=600)
    #plt.show()
    plt.close('all')










#plot 1 param not diff
fname     = cases[key1]['dir']  + '/rads.64.nc'
nc    = netCDF4.Dataset(fname)
ncv   = nc.variables
dates = netCDF4.num2date(ncv['time'][:],ncv['time'].units) 
ind   = np.array(np.where((dates > tim_lim['xmin'] )&(dates<tim_lim['xmax'])  )).squeeze()

for tind in ind:
    print tind
    
    varname   ='radstress_y'
    dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
    val  = dic[varname]
    val [np.abs(val)>1] = 0.0
    date = dic['date']
    #vmin,vmax = val.min()-1e-7,val.max()
    vmin,vmax = -1e-3,1e-3
    
    fig, ax = adcp.make_map()
    fig.set_size_inches(7,7)
    plot_map (ax,tri,val,extent,vmin=vmin,vmax=vmax, dep=dep , cmap = cmap)
    ax.set_title(cases[key1]['label'] + '\n'+varname +' Date: ' + date.isoformat() +'\n Max. Val. =  %.2g' %val.max() + '[m]')
    filename = out_dir1 + '/'+region+ '_' +varname.capitalize()+str(10000+tind)+'_'+date_str+'.png'
    plt.savefig(filename,dpi=600)
    plt.close('all')










lon0 = -180.0
lon1 =  180.0
lat0 = -90.0
lat1 =  90.0
fpng = 'BMNG_hirez.png'

img = plt.imread(fpng)
ax.imshow(img,extent=[lon0,lon1,lat0,lat1],origin='upper')

cf1 = ax.tricontourf(tri,val,levels=levels, cmap = cmap , extend='both')#extend='max' )  #,extend='both'
plt.colorbar(cf1,shrink = 0.5,ticks = [vmin,(vmin+vmax)/2,vmax])      







try: 
  if True:
      for tind in range(1,nt,10):
          #fetch last elev time step
          fname     = base_dir  + '/fort.63.nc'
          varname   ='zeta'
          dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
          val  = dic[varname]
          val [np.isnan(val)] = 0.0
          date = dic['date']
          #vmin,vmax = val.min(),val.max()
          vmin,vmax = -2,2
          save_map(tri,val,date,varname,vmin,vmax)
          plt.close('all') 
except:
    pass

try: 
  if True:
      for tind in range(1,nt,10):
          #fetch last elev time step
          #fname     = base_dir  + '/fort.63.nc'
          #varname   ='zeta'
          #dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
          #elev  = dic[varname]
          #elev [np.isnan(elev)] = 0.0
          
          wdepth = depth
                  
          fname     = base_dir  + '/rads.64.nc'
          varname   ='radstress_y'
          dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
          val  = dic[varname]
          val [np.abs(val)>1] = 0.0
          date = dic['date']
          #vmin,vmax = val.min()-1e-7,val.max()
          vmin,vmax = -1e-2,1e-2
          save_map(tri,val,date,varname,vmin,vmax,wdepth)

          #varname   ='radstress_y'
          #dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
          #val  = dic[varname]
          #val [np.isnan(val)] = 0.0
          #date = dic['date']
          #vmin,vmax = val.min()-1e-7,val.max()
          #vmin,vmax = -2e-2,2e-2
          #save_map(tri,val,date,varname,vmin,vmax,wdepth)

          fname     = base_dir  + '/fort.74.nc'
          varname = 'windy'
          dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
          val  = dic[varname]
          date = dic['date']
          vmin,vmax = -1e-2,1e-2
          #vmin,vmax = val.min()-1e-7,val.max()
          save_map(tri,val,date,varname,vmin,vmax,wdepth)

          plt.close('all') 

except:
    pass

try: 
  if True:
      for tind in range(1,nt,10):
          #fetch last elev time step
          fname     = base_dir  + '/fort.73.nc'
          varname   ='pressure'
          dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
          val  = dic[varname]
          val [np.isnan(val)] = 0.0
          date = dic['date']
          vmin,vmax = -val.max(),val.max()
          #vmin,vmax = 9.4,10

          save_map(tri,val,date,varname,vmin,vmax)
except:
    pass





if False:
  # get number of outputs
  fname     = base_dir  + '/fort.63.nc'
  nt = netCDF4.Dataset(fname).variables['time'][:].shape[0]
  if False:
    plist = np.sort(glob.glob( base_dir  + '/nco_test/fort.63.*.nc'))

    for tind in range(len(plist)):
      try:
        fname = plist[tind]
        print fname 
        ### NOUPC pressure
        varname   ='zeta'
        fname = plist[tind]
        nc   = netCDF4.Dataset(fname) 
        val  = nc.variables[varname][:].squeeze()
        val  = np.ma.masked_where(np.isnan(val),val)
        val [np.isnan(val)] = 0.0
        date = netCDF4.num2date(nc.variables['time'][0],nc.variables['time'].units)
        vmin,vmax = -50,50
        save_map(tri,val,date)
      except:
        pass


if False:
  plist = np.sort(glob.glob( base_dir  + '/field_ocn_pmsl*'))

  for tind in range(len(plist)):
    try:
      fname = plist[tind]
      print fname 
      ### NOUPC pressure
      varname = 'pmsl'
      nc = netCDF4.Dataset(fname) 
      val  = nc.variables[varname][:]
      date = parser.parse(fname[-22:-3])
      #vmin,vmax = val.min(),val.max()
      vmin,vmax = 92000,99000
      save_map(tri,val,date)
      nc.close()
    except:
      pass

if True:
  plist = np.sort(glob.glob( base_dir  + '/field_ocn_sxx*'))

  for tind in range(len(plist)):
    try:
      fname = plist[tind]
      print fname 
      ### NOUPC pressure
      varname = 'sxx'
      nc = netCDF4.Dataset(fname) 
      val  = nc.variables[varname][:]
      date = parser.parse(fname[-22:-3])
      vmin,vmax = val.min(),val.max()
      #vmin,vmax = -30000,30000
      save_map(tri,val,date,varname,vmin,vmax)
      nc.close()
    except:
      pass


if False:
  for tind in range(1,nt):
    #pa2meter_water = 1/9807.
    # plot pressure
    #fname     = base_dir  + '/fort.73.nc'
    #varname = 'pressure'
    #dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
    #val  = dic[varname]
    #date = dic['date']
    #vmin,vmax = 9.8,10.1
    #save_map(tri,val,date)

    #fetch last elev time step
    fname     = base_dir  + '/fort.63.nc'
    varname   ='zeta'
    dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
    val  = dic[varname]
    date = dic['date']
    vmin,vmax = -1,1
    save_map(tri,val,date)

    fname     = base_dir  + '/fort.74.nc'
    varname = 'windx'
    dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
    val  = dic[varname]
    date = dic['date']
    vmin,vmax = -20,20
    save_map(tri,val,date)

    plt.close('all') 


    fname     = base_dir  + '/fort.74.nc'
    varname = 'windy'
    dic  = adcp.ReadVar(fname = fname , varname = varname , time_name = 'time' , tind = tind)
    val  = dic[varname]
    date = dic['date']
    vmin,vmax = -20,20
    save_map(tri,val,date)

    plt.close('all') 

#fname     = base_dir  + '/field_atmesh_pmsl2008-09-03T00:01:00.nc'

if False:
  ### NOUPC wind
  fname     = base_dir  + '/field_ocn_izwh10m2008-09-03T00:01:00.nc'
  varname = 'izwh10m'
  val  = netCDF4.Dataset(fname).variables[varname][:]
  date = parser.parse(fname[-22:-3])
  vmin,vmax = val.min(),val.max()
  save_map(tri,val,date)











#plot_nwm_files
plot_snow = False
if plot_snow:
    land = netCDF4.Dataset(base_info.nwm_results_dir +'/short_range/'+ 'nwm.t01z.short_range.land.f018.conus.nc')
    land_ncv = land.variables
    land_x = land_ncv['x'][:]
    land_y = land_ncv['y'][:]
    land_snow =  land_ncv['SNOWH'][:]
    jj = 20
    plt.pcolor(land_x[::jj],land_y[::jj],land_snow[0,::jj,::jj])





"""

print ('[info]: Fin ')











