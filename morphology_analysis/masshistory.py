import astropy
import astropy.io.fits as pyfits
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.ndimage
import scipy as sp
import illustris_python as ilpy
import h5py
import gfs_sublink_utils as gsu
from astropy.cosmology import WMAP7,z_at_value
import astropy.units as u
import showgalaxy
import illcan_multiplots as icmp
import warnings


ilh = 0.704
illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)
illcos.snaps = np.asarray([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135])

illcos.redshifts = np.asarray([  4.67730470e+01,   4.45622040e+01,   4.24536740e+01,
         4.06395570e+01,   3.87125590e+01,   3.68747400e+01,
         3.51219700e+01,   3.36139400e+01,   3.20120740e+01,
         3.04843400e+01,   2.90273060e+01,   2.76377010e+01,
         2.64421250e+01,   2.51721570e+01,   2.39609610e+01,
         2.28058160e+01,   2.18119640e+01,   2.07562710e+01,
         1.97494330e+01,   1.87891900e+01,   1.79630250e+01,
         1.70854530e+01,   1.62484930e+01,   1.54502670e+01,
         1.47634960e+01,   1.40339920e+01,   1.33382480e+01,
         1.26747020e+01,   1.20418640e+01,   1.14973880e+01,
         1.09190330e+01,   1.03674440e+01,   9.99659050e+00,
         9.84138040e+00,   9.38877130e+00,   9.00233990e+00,
         8.90799920e+00,   8.44947630e+00,   8.01217290e+00,
         7.59510710e+00,   7.23627610e+00,   7.00541700e+00,
         6.85511730e+00,   6.49159770e+00,   6.14490120e+00,
         6.01075740e+00,   5.84661370e+00,   5.52976580e+00,
         5.22758100e+00,   4.99593350e+00,   4.93938070e+00,
         4.66451770e+00,   4.42803370e+00,   4.17683490e+00,
         4.00794510e+00,   3.93726110e+00,   3.70877430e+00,
         3.49086140e+00,   3.28303310e+00,   3.08482260e+00,
         3.00813110e+00,   2.89578500e+00,   2.73314260e+00,
         2.57729030e+00,   2.44422570e+00,   2.31611070e+00,
         2.20792550e+00,   2.10326970e+00,   2.00202810e+00,
         1.90408950e+00,   1.82268930e+00,   1.74357060e+00,
         1.66666960e+00,   1.60423450e+00,   1.53123900e+00,
         1.47197480e+00,   1.41409820e+00,   1.35757670e+00,
         1.30237850e+00,   1.24847260e+00,   1.20625810e+00,
         1.15460270e+00,   1.11415060e+00,   1.07445790e+00,
         1.03551040e+00,   9.97294230e-01,   9.87852810e-01,
         9.50531350e-01,   9.23000820e-01,   8.86896940e-01,
         8.51470900e-01,   8.16709980e-01,   7.91068250e-01,
         7.57441370e-01,   7.32636180e-01,   7.00106350e-01,
         6.76110410e-01,   6.44641840e-01,   6.21428750e-01,
         5.98543290e-01,   5.75980850e-01,   5.46392180e-01,
         5.24565820e-01,   5.03047520e-01,   4.81832940e-01,
         4.60917790e-01,   4.40297850e-01,   4.19968940e-01,
         3.99926960e-01,   3.80167870e-01,   3.60687660e-01,
         3.47853840e-01,   3.28829720e-01,   3.10074120e-01,
         2.91583240e-01,   2.73353350e-01,   2.61343260e-01,
         2.43540180e-01,   2.25988390e-01,   2.14425040e-01,
         1.97284180e-01,   1.80385260e-01,   1.69252030e-01,
         1.52748770e-01,   1.41876200e-01,   1.25759330e-01,
         1.09869940e-01,   9.94018030e-02,   8.38844310e-02,
         7.36613850e-02,   5.85073230e-02,   4.85236300e-02,
         3.37243720e-02,   2.39744280e-02,   9.52166700e-03,
                              0.00])

illcos.ages = np.asarray( illcos.age(illcos.redshifts) )

def mergerfileinfo(snapkey,subfindID,size=2,trange=[-2.0,2.0]):
    #merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo_SB25.hdf5'
    merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo_SB25_2017March3.hdf5'


    #update figure and exploratory quantities here
    #want to build training sets cleverly from these parameters
    
    with h5py.File(merger_file,'r') as mcat:
        vals = mcat['mergerinfo'][snapkey]['latest_NumMajorMergersLastGyr'].value
        sfids = mcat['mergerinfo'][snapkey]['SubfindID'].value
        lsn = mcat['mergerinfo'][snapkey]['LatestTree_snapnum'].value
        last_merger = mcat['mergerinfo'][snapkey]['latest_SnapNumLastMajorMerger'].value
        last_minmerger = mcat['mergerinfo'][snapkey]['latest_SnapNumLastMinorMerger'].value

        
        sfi = np.where(sfids==subfindID)[0][0]
              
        val=vals[sfi]
        

    #also: get ALL keys for this subhalo
    return gsu.age_at_snap(lsn), gsu.age_at_snap(last_merger[sfi]), gsu.age_at_snap(last_minmerger[sfi])


def masshistory(snapkey,subfindID,camnum=0,basepath='/astro/snyder_lab2/Illustris/Illustris-1',size=2,trange=[-1.5,1.5],alph=0.2,Q=8.0,sb='SB25',gyrbox=True,filters=['NC-F115W','NC-F150W','NC-F200W'],use_rfkey=True,npix=None,radeff=0.2,dx=0,dy=0,savefile=None):
    snapnum = snapkey[-3:]
    this_snap_int = np.int64(snapnum)

    sublink_id,tree_id,mstar,sfr = gsu.sublink_id_from_subhalo(basepath,this_snap_int,subfindID)
    tree = gsu.load_full_tree(basepath,tree_id)
    mmpb_sublink_id,snaps,mass,mstar,r1,r2,mmpb_sfid,sfr,times,mlpids,bhmdot,bhm = gsu.mmpb_from_tree(tree,sublink_id)
    time_now = gsu.age_at_snap(this_snap_int)


    time_latest,time_lastmajor,time_lastminor = mergerfileinfo(snapkey,subfindID)
    print(time_latest,time_lastmajor,time_lastminor)

    bhlum = radeff*((u.Msun/u.year)*bhmdot*((1.0e10)/ilh)/(0.978*1.0e9/ilh))*(astropy.constants.c**2)
    bhlum_lsun = bhlum.to('solLum')

    bhm_msun = bhm*(1.0e10)/ilh
    
    fig=pyplot.figure(figsize=(size*2,size)) 
    pyplot.subplots_adjust(wspace=0.0,hspace=0.0)
    
    #mass history plot
    axi = fig.add_subplot(1,2,1)

    axi.semilogy(times-time_now,mstar)
    
    axi.semilogy(times[bhm_msun > 0]-time_now,bhm_msun[bhm_msun > 0]*2.0e2)

    axi.legend(['$M_* (t)$','$M_{bh} (t) x200$'],loc='upper left',fontsize=25)
    
    axi.plot([0.0,0.0],[1.0e8,1.0e12],marker=None,linestyle='dashed',color='black',linewidth=2.0)

    if gyrbox is True:
        axi.plot([time_latest-time_now,time_latest-time_now],[1.0e8,1.0e13],marker=None,linestyle='solid',color='gray',linewidth=1.0)
        axi.plot([time_latest-time_now-1.0,time_latest-time_now-1.0],[1.0e8,1.0e13],marker=None,linestyle='solid',color='gray',linewidth=1.0)

    axi.plot([time_lastmajor-time_now,time_lastmajor-time_now],[1.0e8,1.0e13],marker=None,linestyle='solid',color='Red',linewidth=4.0)
    axi.plot([time_lastminor-time_now,time_lastminor-time_now],[1.0e8,1.0e13],marker=None,linestyle='dotted',color='Red',linewidth=4.0)

    axi.set_xlim(trange[0],trange[1])
    inrange=np.where(np.logical_and(times-time_now > trange[0], times-time_now <= trange[1]))[0]
    axi.set_ylim(np.min(mstar[inrange])/2.0,np.max(mstar[inrange])*2.0)

    ls = 25
    axi.set_xlabel('$t-t_{obs} (Gyr)$',size=ls)
    #axi.set_ylabel('$M_* (t)$,   $M_{bh} (t)/200$',size=ls)
    axi.tick_params(axis='both', which='major', labelsize=ls)
    
    #image(s?)
    axi = fig.add_subplot(1,2,2)

    camstr='{:02d}'.format(camnum)

    im_snap_keys, im_rf_fil_keys, im_npix = icmp.return_rf_variables()

    if use_rfkey is True:
        try:
            rfkey=im_rf_fil_keys[im_snap_keys==snapkey][0]
            npix=im_npix[im_snap_keys==snapkey][0]
        except:
            rfkey=None
            npix=400
    else:
        rfkey=None
        if npix is None:
            npix=400
    
    axi = showgalaxy.showgalaxy(axi,snapkey,subfindID,camstr,rfkey=rfkey,Npix=npix,alph=alph,Q=Q,sb=sb,filters=filters,dx=dx,dy=dy)

    if savefile is None:
        pyplot.show()
    else:
        fig.savefig(savefile,dpi=500)

    pyplot.close(fig)
    
    
    return 0







def avail():
    stuff = ['LatestTree_SFID', 'LatestTree_snapnum', 'MainLeafID_match', 'SnapNums', 'SubfindID', 'Tree_SFID_grid', 'latest_BaryonicMassFromMajorMergers', 'latest_BaryonicMassFromMinorMergers', 'latest_NumMajorMergersLastGyr', 'latest_NumMajorMergersLastGyrBaryonic', 'latest_NumMajorMergersSinceRedshiftOne', 'latest_NumMajorMergersSinceRedshiftOneBaryonic', 'latest_NumMajorMergersSinceRedshiftThree', 'latest_NumMajorMergersSinceRedshiftThreeBaryonic', 'latest_NumMajorMergersSinceRedshiftTwo', 'latest_NumMajorMergersSinceRedshiftTwoBaryonic', 'latest_NumMajorMergersTotal', 'latest_NumMajorMergersTotalBaryonic', 'latest_NumMinorMergersLastGyr', 'latest_NumMinorMergersLastGyrBaryonic', 'latest_NumMinorMergersSinceRedshiftOne', 'latest_NumMinorMergersSinceRedshiftOneBaryonic', 'latest_NumMinorMergersSinceRedshiftThree', 'latest_NumMinorMergersSinceRedshiftThreeBaryonic', 'latest_NumMinorMergersSinceRedshiftTwo', 'latest_NumMinorMergersSinceRedshiftTwoBaryonic', 'latest_NumMinorMergersTotal', 'latest_NumMinorMergersTotalBaryonic', 'latest_SnapNumLastMajorMerger', 'latest_SnapNumLastMajorMergerBaryonic', 'latest_SnapNumLastMinorMerger', 'latest_SnapNumLastMinorMergerBaryonic', 'latest_StellarMassFromMajorMergers', 'latest_StellarMassFromMinorMergers', 'merger_span_gyr', 'this_BaryonicMassFromMajorMergers', 'this_BaryonicMassFromMinorMergers', 'this_NumMajorMergersLastGyr', 'this_NumMajorMergersLastGyrBaryonic', 'this_NumMajorMergersSinceRedshiftOne', 'this_NumMajorMergersSinceRedshiftOneBaryonic', 'this_NumMajorMergersSinceRedshiftThree', 'this_NumMajorMergersSinceRedshiftThreeBaryonic', 'this_NumMajorMergersSinceRedshiftTwo', 'this_NumMajorMergersSinceRedshiftTwoBaryonic', 'this_NumMajorMergersTotal', 'this_NumMajorMergersTotalBaryonic', 'this_NumMinorMergersLastGyr', 'this_NumMinorMergersLastGyrBaryonic', 'this_NumMinorMergersSinceRedshiftOne', 'this_NumMinorMergersSinceRedshiftOneBaryonic', 'this_NumMinorMergersSinceRedshiftThree', 'this_NumMinorMergersSinceRedshiftThreeBaryonic', 'this_NumMinorMergersSinceRedshiftTwo', 'this_NumMinorMergersSinceRedshiftTwoBaryonic', 'this_NumMinorMergersTotal', 'this_NumMinorMergersTotalBaryonic', 'this_SnapNumLastMajorMerger', 'this_SnapNumLastMajorMergerBaryonic', 'this_SnapNumLastMinorMerger', 'this_SnapNumLastMinorMergerBaryonic', 'this_StellarMassFromMajorMergers', 'this_StellarMassFromMinorMergers']
    return
