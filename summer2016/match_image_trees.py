import cProfile
import pstats
import math
import string
import sys
import struct
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as pyplot
import matplotlib.colors as pycolors
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import cPickle
import asciitable
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import scipy.odr as odr
import glob
import os
import make_color_image
import make_fake_wht
import gzip
import tarfile
import shutil
import cosmocalc
import congrid
import astropy.io.ascii as ascii
import warnings
import subprocess
import photutils
import astropy
import astropy.cosmology
import astropy.io.fits as pyfits
import astropy.units as u
from astropy.constants import G
from astropy.cosmology import WMAP7,z_at_value
from astropy.coordinates import SkyCoord
import copy
import medianstats_bootstrap as msbs
import illustris_python as ilpy
import h5py
from parse_illustris_morphs import *
from PyML import machinelearning as pyml
from PyML import convexhullclassifier as cvx
import gfs_sublink_utils as gsu

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

def get_merger_quantities(t,times,snaps,sfids,mass,mstar,sfr,snap,sfid,time_span=0.5,basepath='/astro/snyder_lab2/Illustris/Illustris-1'):

    i_span = np.where( np.logical_and( times <= t+time_span, times >= t-time_span))[0]
    i_latest = np.argmax(times[i_span]) #index into i_span
    snap_latest = snaps[i_span[i_latest]]
    sfid_latest = sfids[i_span[i_latest]]


    return snap_latest, sfid_latest


def match_progdesc(snap, sfid, snapnums_int, latest_snap, latest_span=0.5,basepath='/astro/snyder_lab2/Illustris/Illustris-1'):
    this_snap_int = np.int32(snap[-3:])

    snaps_int = snapnums_int #np.int64(np.asarray(snaplist))

    sublink_id,tree_id,mstar,sfr = gsu.sublink_id_from_subhalo(basepath,this_snap_int,sfid)
    tree = gsu.load_full_tree(basepath,tree_id)
    mmpb_sublink_id,snaps,mass,mstar,r1,r2,mmpb_sfid,sfr,times = gsu.mmpb_from_tree(tree,sublink_id)
    time_now = gsu.age_at_snap(this_snap_int)

    #snap_latest,sfid_latest = get_merger_quantities(time_now,times,snaps,mmpb_sfid,mass,mstar,sfr,this_snap_int,sfid,time_span=latest_span)
    #print snaps, latest_snap

    lsi = np.where(snaps==latest_snap)[0]
    sfid_latest = mmpb_sfid[lsi]

    print time_now, latest_snap, sfid_latest, lsi

    matched_mmpb = np.zeros_like(snaps_int)
    matched_sfid = np.zeros_like(snaps_int)

    for i,snap in enumerate(snaps_int):
        tsi = np.where(snaps==snap)[0]
        if tsi.shape[0] == 1:
            matched_mmpb[i]=mmpb_sublink_id[tsi]
            matched_sfid[i]=mmpb_sfid[tsi]
        else:
            matched_mmpb[i] = -1
            matched_sfid[i] = -1

    return matched_mmpb, matched_sfid, sublink_id, latest_snap, sfid_latest


if __name__=="__main__":

    bp = '/astro/snyder_lab2/Illustris/Illustris-1'  #simulation base path for merger trees
    basepath = bp

    catalogfile = "/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB27_12filters_all_NEW.hdf5"   #morphology catalog file
    mergerdatafile = "/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo_SB27.hdf5"

    mdf = h5py.File(mergerdatafile,'w')
    grp = mdf.create_group('mergerinfo')
    
    sfid_dict = {}

    morphcat = h5py.File(catalogfile)
    topkey = morphcat.keys()[0]
    snapkeys = morphcat[topkey].keys()
    print len(snapkeys)

    last_sfid_grid = None

    merger_span = 0.5 #gyr range to forward-search for mergers

    snapnums_int = np.zeros(len(snapkeys),dtype=np.int32)
    for i,sk in enumerate(snapkeys):
        sni = np.int32(sk[-3:])
        snapnums_int[i]=sni

    snapnums_int = np.append(snapnums_int,135)
    
    for s,snapkey in enumerate(snapkeys):
        print snapkey
        print morphcat[topkey][snapkey].keys()
        subfind_id = morphcat[topkey][snapkey]['SubfindID'].value
        print subfind_id.shape

        sgrp = grp.create_group(snapkey)
        sgrp.create_dataset('SubfindID',data=np.int64(subfind_id))
        
        this_snap_treesfid = np.zeros((subfind_id.shape[0],len(snapnums_int)),dtype=np.int64 )-1
        latest_snap_treesfid = np.zeros((subfind_id.shape[0]),dtype=np.int64 )-1

        print this_snap_treesfid.shape

        this_snap_int = np.int32(snapkey[-3:])
        this_snap_time = gsu.age_at_snap(this_snap_int)
        latest_time = this_snap_time + merger_span
        lti = np.where(illcos.ages <= latest_time)[0]
        latest_i = lti[-1]
        latest_snap = illcos.snaps[latest_i]
        if latest_snap==55:
            latest_snap=56
            latest_time = gsu.age_at_snap(latest_snap)

        print latest_snap, latest_i, latest_time, this_snap_time, this_snap_int


        for i,sfid in enumerate(subfind_id):
            existing_line=None
            if last_sfid_grid is not None:
                tbi = np.where(last_sfid_grid[:,s]==sfid)[0]
                if tbi.shape[0]==1:
                    existing_line = last_sfid_grid[tbi[0],:]
            if existing_line is not None:
                this_snap_treesfid[i,:]=existing_line
                print 'line exists:', sfid,existing_line
            else:
                slid_grid,sfid_grid,sublink_id,snap_latest,sfid_latest = match_progdesc(snapkey,sfid,snapnums_int,latest_snap)
                print 'new line: ', sfid,sfid_grid
                this_snap_treesfid[i,:]=sfid_grid
                if sfid_latest.shape[0]==1:
                    latest_snap_treesfid[i]=sfid_latest
                else:
                    latest_snap_treesfid[i]=-1

                    
        sfid_dict[snapkey]=this_snap_treesfid
        last_sfid_grid = this_snap_treesfid

        sgrp.create_dataset('SnapNums',data=np.int32(snapnums_int))
        sgrp.create_dataset('Tree_SFID_grid',data=np.int64(this_snap_treesfid))
        sgrp.create_dataset('LatestTree_SFID',data=np.int64(latest_snap_treesfid))
        sgrp.create_dataset('merger_span_gyr',data=merger_span)
        sgrp.create_dataset('LatestTree_snapnum',data=latest_snap)

        
        merger_history_dir = os.path.join(basepath,'MERGER_HISTORY')
        merger_file_this = os.path.join(merger_history_dir,'merger_history_{:03d}'.format(this_snap_int)+'.hdf5')
        merger_file_latest = os.path.join(merger_history_dir,'merger_history_{:03d}'.format(latest_snap)+'.hdf5')

        merger_dict = {}
        merger_cat_this = h5py.File(merger_file_this)
        merger_cat_latest = h5py.File(merger_file_latest)
    
        keys = merger_cat_this.keys()
        for f in keys:
            merger_dict['this_'+f]=(merger_cat_this[f].value)[subfind_id]
            merger_dict['latest_'+f]=(merger_cat_latest[f].value)[latest_snap_treesfid]

            sgrp.create_dataset('this_'+f,data=(merger_cat_this[f].value)[subfind_id])
            sgrp.create_dataset('latest_'+f,data=(merger_cat_latest[f].value)[latest_snap_treesfid])
            print 'latest_'+f, merger_dict['latest_'+f][0:10]




    mdf.close()
    
    z1key = 'snapshot_085'
    catalogkeys = morphcat[topkey][z1key].keys()
    stellar_mass = morphcat[topkey][z1key]['Mstar_Msun'].value
    sfr = morphcat[topkey][z1key]['SFR_Msunperyr'].value
    filters = morphcat[topkey][z1key]['Filters'].value
    print filters

    morphkeys = morphcat[topkey][z1key]['WFC3-F160W']['CAMERA1'].keys()
    print morphkeys


    halflightradius_Hband_z1_cam1 = morphcat[topkey][z1key]['WFC3-F160W']['CAMERA1']['RHALF'].value
