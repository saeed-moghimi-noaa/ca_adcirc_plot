# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:24:14 2017

@author: Sergey.Vinogradov
"""
from datetime import datetime
#==============================================================================
def readTrack ( atcfFile ):
    """
    Reads ATCF-formatted file
    Args:
        'atcfFile': (str) - full path to the ATCF file
    Returns:
        dict: 'lat', 'lon', 'vmax', 'mslp','dates'
    """
    lines = open(atcfFile).readlines()
        
    myOcn  = []
    myCy   = []
    myDate = []
    myLat  = []
    myLon  = []
    myVmax = []
    myMSLP = []
    for line in lines:
        r = line.rstrip().split(',')
        myOcn.append  (r[0])
        myCy.append   (int(r[1]))
        myDate.append (datetime.strptime(r[2].strip(),'%Y%m%d%H'))
        latSign = -1.0
        if 'N' in r[6]:
            latSign = 1.0     
        myLat.append  (latSign*0.1*float(r[6][:-1]))
        lonSign = -1.0
        if 'E' in r[7]:
            lonSign = 1.0
        myLon.append  (lonSign*0.1*float(r[7][:-1]))
        myVmax.append (float(r[8]))
        myMSLP.append (float(r[9]))
    
    return { 
            'basin' : myOcn,    'cy' : myCy, 'dates' : myDate, 
            'lat'   : myLat,   'lon' : myLon,
            'vmax'  : myVmax, 'mslp' : myMSLP }

#==============================================================================
if __name__ == "__main__":
    
    # Demo    
    atcfFile = 'bal182012.dat'
    trk = readTrack (atcfFile)
    
    import matplotlib.pyplot as plt
    plt.plot(trk['lon'],trk['lat'])
    