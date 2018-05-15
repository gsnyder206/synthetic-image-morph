import cProfile
import pstats
import math
import string
import sys
import struct
import numpy as np
import asciitable
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import scipy.odr as odr
import glob
import os
#import make_color_image
#import make_fake_wht
import gzip
import tarfile
import shutil
#import cosmocalc
import congrid
import astropy.io.ascii as ascii
import warnings
import subprocess
#import photutils
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

ilh = 0.704
illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)

def redshift_from_snapshot(snapnum):
    #snapshot z lookup table
    snaps = np.asarray([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
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

    redshifts = np.asarray([  4.67730470e+01,   4.45622040e+01,   4.24536740e+01,
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

    si = np.where(snaps==snapnum)[0]
    if si.shape[0] != 1:
        return -1
    else:
        return np.float32(redshifts[si])[0]



def age_at_z(z):
    #use illustris cosmology
    illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)
    t_gyr = np.asarray(illcos.age(z))
    return t_gyr

def age_at_snap(snapnum):
    z_at_snap = redshift_from_snapshot(snapnum)
    t_gyr = -1.0
    if z_at_snap != -1:
        t_gyr = age_at_z(z_at_snap)

    return np.float32(t_gyr)

def load_full_tree(basepath,tree_id):
    this_file = return_tree_file(basepath,tree_id)

    result = {'id':tree_id}

    fields = ['TreeID','SubhaloID','SnapNum','SubfindID','DescendantID','FirstProgenitorID','MassHistory','RootDescendantID','SubhaloMass','SubhaloMassInRadType','SubhaloMassInHalfRadType','SubhaloSFR','SubhaloParent','SubhaloPos','SubhaloVel','SubhaloStellarPhotometricsMassInRad','MainLeafProgenitorID','SubhaloBHMdot','SubhaloBHMass']

    with h5py.File(this_file,'r') as f:
        tree_id_values = f['TreeID'].value
        tree_index = np.where(tree_id_values==tree_id)[0]
        tree_start = np.min(tree_index)
        tree_n = tree_index.shape

        if fields is None:
            fields = f.keys()
        for field in fields:
            result[field] = f[field][tree_start:tree_start+tree_n]

    return result

def return_tree_file(basepath,tree_id):
    chunkno = np.int64(tree_id/1e16) #assuming this scheme for file splitting
    treef = ilpy.sublink.treePath(basepath,chunkNum=chunkno)

    return treef

def sublink_id_from_subhalo(basepath,snap,sfid,gal=False):

    offsets_dir = os.path.join(os.path.join(os.path.join(basepath,'trees'),'SubLink'),'offsets')
    if os.path.lexists(offsets_dir):
        gal=True

    fields = ['SubhaloID','TreeID','SubhaloMass','SubhaloMassInRadType','SnapNum','SubhaloSFR']
    tree = ilpy.sublink.loadTree(basepath,snap,sfid,fields=fields,onlyMPB=True,gal=gal)
    sublink_id = np.int64(tree['SubhaloID'][0])
    tree_id = np.int64(tree['TreeID'][0])

    mstar = np.float32(tree['SubhaloMassInRadType'][0,4])
    sfr = np.float32(tree['SubhaloSFR'][0])

    return sublink_id, tree_id, mstar, sfr


def progmass_from_subhalo_at_snap(basepath,snap,sfid,othersnap,gal=False):
    offsets_dir = os.path.join(os.path.join(os.path.join(basepath,'trees'),'SubLink'),'offsets')
    if os.path.lexists(offsets_dir):
        gal=True

    fields = ['SubhaloID','TreeID','SubhaloMass','SubhaloMassInRadType','SnapNum']
    tree = ilpy.sublink.loadTree(basepath,snap,sfid,fields=fields,onlyMPB=True,gal=gal)
    sublink_id = np.int64(tree['SubhaloID'][0])
    tree_id = np.int64(tree['TreeID'][0])

    snaplist = np.int32(tree['SnapNum'])
    si = np.where(snaplist==othersnap)[0]
    if si.shape[0]==0:
        return -1.0
    else:
        massinradtype = np.int64(tree['SubhaloMassInRadType'][si,4])
        return massinradtype

    return sublink_id


def descendants_from_tree(tree,sublink_id):
    sl_ind = np.where(tree['SubhaloID']==sublink_id)[0]
    if sl_ind.shape[0] == 0:
        return -1, -1

    rootid = np.int64(tree['RootDescendantID'][sl_ind[0]])
    mlpid = np.int64(tree['MainLeafProgenitorID'][sl_ind[0]])
    
    desc = [sublink_id]
    snap = [tree['SnapNum'][sl_ind[0]]]
    mass = [tree['SubhaloMass'][sl_ind[0]]]
    mstar = [tree['SubhaloMassInRadType'][sl_ind[0],4]]
    #mstar = [tree['SubhaloStellarPhotometricsMassInRad'][sl_ind[0]]]
    sfid = [tree['SubfindID'][sl_ind[0]]]
    sfr = [tree['SubhaloSFR'][sl_ind[0]]]
    mlpid = [tree['MainLeafProgenitorID'][sl_ind[0]]]
    bhmdot = [tree['SubhaloBHMdot'][sl_ind[0]]]
    bhm = [tree['SubhaloBHMass'][sl_ind[0]]]
    
    desc_sli = sublink_id

    while desc_sli != -1:
        desc_sli = tree['DescendantID'][sl_ind[0]]
        if desc_sli != -1:
            sl_ind = np.where(tree['SubhaloID']==desc_sli)[0]
            desc.append(desc_sli)
            snap.append(tree['SnapNum'][sl_ind])
            mass.append(tree['SubhaloMass'][sl_ind])
            mstar.append(tree['SubhaloMassInRadType'][sl_ind,4])
            sfid.append(tree['SubfindID'][sl_ind])
            sfr.append(tree['SubhaloSFR'][sl_ind])
            mlpid.append(tree['MainLeafProgenitorID'][sl_ind])
            bhmdot.append(tree['SubhaloBHMdot'][sl_ind])
            bhm.append(tree['SubhaloBHMass'][sl_ind])
            
    return np.asarray(desc,dtype='int64'),np.asarray(snap),rootid,np.asarray(mass)*(1.0e10)/ilh,np.asarray(mstar)*(1.0e10)/ilh, np.asarray(sfid), np.asarray(sfr), np.asarray(mlpid), np.asarray(bhmdot), np.asarray(bhm)


def mainprogs_from_tree(tree,sublink_id):
    sl_ind = np.where(tree['SubhaloID']==sublink_id)[0]
    if sl_ind.shape[0] == 0:
        return -1, -1

    rootid = np.int64(tree['RootDescendantID'][sl_ind[0]])
    mlpid = np.int64(tree['MainLeafProgenitorID'][sl_ind[0]])

    desc = [sublink_id]
    snap = [tree['SnapNum'][sl_ind[0]]]
    mass = [tree['SubhaloMass'][sl_ind[0]]]
    mstar = [tree['SubhaloMassInRadType'][sl_ind[0],4]]
    #mstar = [tree['SubhaloStellarPhotometricsMassInRad'][sl_ind[0]]]
    sfid = [tree['SubfindID'][sl_ind[0]]]
    sfr = [tree['SubhaloSFR'][sl_ind[0]]]
    mlpid = [tree['MainLeafProgenitorID'][sl_ind[0]]]
    bhmdot = [tree['SubhaloBHMdot'][sl_ind[0]]]
    bhm = [tree['SubhaloBHMass'][sl_ind[0]]]

    desc_sli = sublink_id

    while desc_sli != -1:
        desc_sli = tree['FirstProgenitorID'][sl_ind[0]]
        if desc_sli != -1:
            sl_ind = np.where(tree['SubhaloID']==desc_sli)[0]
            desc.append(desc_sli)
            snap.append(tree['SnapNum'][sl_ind])
            mass.append(tree['SubhaloMass'][sl_ind])
            mstar.append(tree['SubhaloMassInRadType'][sl_ind,4])
            sfid.append(tree['SubfindID'][sl_ind])
            sfr.append(tree['SubhaloSFR'][sl_ind])
            mlpid.append(tree['MainLeafProgenitorID'][sl_ind])
            bhmdot.append(tree['SubhaloBHMdot'][sl_ind])
            bhm.append(tree['SubhaloBHMass'][sl_ind])

    return np.asarray(desc,dtype='int64'),np.asarray(snap),rootid,np.asarray(mass)*(1.0e10)/ilh,np.asarray(mstar)*(1.0e10)/ilh, np.asarray(sfid), np.asarray(sfr), np.asarray(mlpid), np.asarray(bhmdot), np.asarray(bhm)


def mmpb_from_tree(tree,sublink_id):

    desc1,snap1,rootid1,mass1,mstar1,sfid1,sfr1,mlpid1,bhmdot1,bhm1 = descendants_from_tree(tree,sublink_id)
    desc2,snap2,rootid2,mass2,mstar2,sfid2,sfr2,mlpid2,bhmdot2,bhm2 = mainprogs_from_tree(tree,sublink_id)
    
    sli = np.concatenate((np.flipud(desc2),desc1[1:]))
    snap = np.concatenate((np.flipud(snap2),snap1[1:]))
    mass = np.concatenate((np.flipud(mass2),mass1[1:]))
    mstar = np.concatenate((np.flipud(mstar2),mstar1[1:]))
    sfid = np.concatenate((np.flipud(sfid2),sfid1[1:]))
    sfr = np.concatenate((np.flipud(sfr2),sfr1[1:]))
    mlpid = np.concatenate((np.flipud(mlpid2),mlpid1[1:]))
    bhmdot = np.concatenate((np.flipud(bhmdot2),bhmdot1[1:]))
    bhm = np.concatenate((np.flipud(bhm2),bhm1[1:]))

    times = np.zeros_like(mstar)
    for i,sn in enumerate(snap):
        times[i]=age_at_snap(sn)

    return sli,snap,mass,mstar,rootid1,rootid2,sfid,sfr,times,mlpid,bhmdot,bhm


def evaluate_primary(basepath,primary_snap,primary_sfid,ratio = 4.0):
    hasMajorMerger = False
    merger_snapshot = -1
    time_until_merger = -1.0
    time_of_merger = -1.0
    merger_redshift = -1.0
    merger_mass = -1.0
    merger_mstar = -1.0
    sublink_id_primary,tree_id_primary,primary_mstar = sublink_id_from_subhalo(basepath,primary_snap,primary_sfid)
    tree = load_full_tree(basepath,tree_id_primary)

    tree_mstar = tree['SubhaloMassInRadType'][:,4]

    primary_desc,primary_snaps,primary_root,primary_mass,primary_mstar = descendants_from_tree(tree,sublink_id_primary)
    for i,pdi in enumerate(primary_desc):
        snap = primary_snaps[i]
        progs = np.where(np.logical_and(tree['DescendantID']==pdi, tree['SnapNum']==snap-1))[0]
        
        if progs.shape[0] > 1:
            spi = np.argsort(tree_mstar[progs])
            mass_ratio = tree_mstar[progs[spi[-2]]]/tree_mstar[progs[spi[-1]]]
            print(snap, primary_mstar[i], tree_mstar[progs[spi[-1]]], tree_mstar[progs[spi[-2]]])
            if mass_ratio > 1.0/ratio:
               return True, snap, age_at_snap(snap) - age_at_snap(primary_snap), age_at_snap(snap), redshift_from_snap(snap), tree['SubhaloMass'][pdi], tree['SubhaloMassInRadType'][pdi,4]

        else:
            print(snap, primary_mstar[i])

    return hasMajorMerger, merger_snapshot, time_until_merger, time_of_merger, merger_redshift, merger_mass, merger_mstar




def evaluate_pair(basepath,primary_snap,primary_sfid,secondary_snap,secondary_sfid):

    pairMerges = False
    merger_snapshot = -1
    time_until_merger = -1.0
    time_of_merger = -1.0
    merger_redshift = -1.0
    merger_mass = -1.0
    merger_mstar = -1.0
    mass_ratio_tmax = -1.0
    mass_ratio_now = -1.0

    mhalo_ratio_tmax = -1.0
    mhalo_ratio_now = -1.0

    sec_snap_tmax = -1
    
    pri_sfr = -1
    sec_sfr = -1

    #if these two subhalos merge, then they must be contained in the same tree
    try:
        sublink_id_primary,tree_id_primary,primary_mstar,pri_sfr = sublink_id_from_subhalo(basepath,primary_snap,primary_sfid)
        sublink_id_secondary,tree_id_secondary,secondary_mstar,sec_sfr = sublink_id_from_subhalo(basepath,secondary_snap,secondary_sfid)
    except ValueError as e:
        print(e)
        return pairMerges, merger_snapshot, time_until_merger, time_of_merger, merger_redshift, merger_mass, merger_mstar, mass_ratio_tmax,mass_ratio_now,sec_snap_tmax,mhalo_ratio_tmax, pri_sfr, sec_sfr

    mass_ratio_now = secondary_mstar/primary_mstar

    if True:
        primary_tree = load_full_tree(basepath,tree_id_primary)
        secondary_tree = load_full_tree(basepath,tree_id_secondary)

        primary_desc,primary_snaps,primary_mass,primary_mstar,r1,r2,sfidA,sfrA,timesA,mlpidsA,bhmdotA,bhmA = mmpb_from_tree(primary_tree,sublink_id_primary)
        secondary_desc,secondary_snaps,secondary_mass,secondary_mstar,r1s,r2s,sfidB,sfrB,timesB,mlpidsB,bhmdotB,bhmB = mmpb_from_tree(secondary_tree,sublink_id_secondary)

        for i,pdi in enumerate(primary_desc):
            matchi = np.where(np.logical_and(secondary_desc==pdi,secondary_snaps > secondary_snap))[0]

            if matchi.shape[0] > 0:
                pairMerges = True

                first_match = matchi[np.argmin(secondary_snaps[matchi])]
                merger_snapshot = secondary_snaps[first_match]
                merger_mass = secondary_mass[first_match]
                merger_mstar = secondary_mstar[first_match]
                merger_redshift = redshift_from_snapshot(merger_snapshot)
                time_until_merger = age_at_snap(merger_snapshot) - age_at_snap(primary_snap)
                time_of_merger = age_at_snap(merger_snapshot)

                sec_mstar_hist = secondary_mstar[0:matchi]
                sec_mhalo_hist = secondary_mass[0:matchi]

                sec_snap_hist = secondary_snaps[0:matchi]

                mi = np.argmax(sec_mstar_hist)
                sec_mstar_tmax = sec_mstar_hist[mi]
                sec_mhalo_tmax = sec_mhalo_hist[mi]

                sec_snap_tmax = sec_snap_hist[mi]
                pi = np.where(primary_snaps==sec_snap_tmax)[0]
                if pi.shape[0] == 1:
                    pri_mstar_tmax = primary_mstar[pi]
                    pri_mhalo_tmax = primary_mass[pi]
                    
                    mass_ratio_tmax = sec_mstar_tmax/pri_mstar_tmax
                    mhalo_ratio_tmax = sec_mhalo_tmax/pri_mhalo_tmax
                    
                    mass_ratio_tmax = np.float32(mass_ratio_tmax[0])
                    mhalo_ratio_tmax = np.float32(mhalo_ratio_tmax[0])

            
                break

        if pairMerges is False:

            sec_mstar_hist = secondary_mstar#[0:matchi]
            sec_mhalo_hist = secondary_mass#[0:matchi]

            sec_snap_hist = secondary_snaps#[0:matchi]

            mi = np.argmax(sec_mstar_hist)
            sec_mstar_tmax = sec_mstar_hist[mi]
            sec_mhalo_tmax = sec_mhalo_hist[mi]
            sec_snap_tmax = sec_snap_hist[mi]

            pi = np.where(primary_snaps==sec_snap_tmax)[0]
            if pi.shape[0] == 1:
                pri_mstar_tmax = primary_mstar[pi]
                pri_mhalo_tmax = primary_mass[pi]
                    
                mass_ratio_tmax = sec_mstar_tmax/pri_mstar_tmax
                mhalo_ratio_tmax = sec_mhalo_tmax/pri_mhalo_tmax
                
                mass_ratio_tmax = np.float32(mass_ratio_tmax[0])
                mhalo_ratio_tmax = np.float32(mhalo_ratio_tmax[0])
                


        

    return pairMerges, merger_snapshot, time_until_merger, time_of_merger, merger_redshift, merger_mass, merger_mstar, mass_ratio_tmax,mass_ratio_now,sec_snap_tmax,mhalo_ratio_tmax,pri_sfr, sec_sfr




def parse_pair_catalog(pairfile):
    # primary snapid subhaloid iz tz mstar mhalo mbary ncompanions
    # secondary  snapid subhaloid dradius  iz tz mstar  mhalo mbary

    isPrimary = True

    pri_snap = []
    pri_sfid = []
    pri_iz = []
    pri_tz = []
    pri_mstar = []
    pri_mhalo = []
    pri_mbary = []
    pri_nc = []
    
    sec_snap = []
    sec_sfid = []
    sec_drad_arcsec = []
    sec_iz = []
    sec_tz = []
    sec_mstar = []
    sec_mhalo = []
    sec_mbary = []
    sec_nc = []    


    sc = 0

    with open(pairfile,'r') as pf:
        for i,line in enumerate(pf):
            if i < 4:
                continue
            data = line.split()

            if isPrimary:
                primary_data = copy.copy(data)
                Nc = int(data[7])
                isPrimary=False
            else:
                sc = sc + 1
                #here's a pair, do stuff
                pri_snap.append(primary_data[0])
                pri_sfid.append(primary_data[1])
                pri_iz.append(primary_data[2])
                pri_tz.append(primary_data[3])
                pri_mstar.append(primary_data[4])
                pri_mhalo.append(primary_data[5])
                pri_mbary.append(primary_data[6])
                pri_nc.append(primary_data[7])

                sec_snap.append(data[0])
                sec_sfid.append(data[1])
                sec_drad_arcsec.append(data[2])
                sec_iz.append(data[3])
                sec_tz.append(data[4])
                sec_mstar.append(data[5])
                sec_mhalo.append(data[6])
                sec_mbary.append(data[7])
                
                print(sc, Nc, pri_snap[-1], pri_sfid[-1], sec_snap[-1], sec_sfid[-1])

                if sc==Nc:
                    isPrimary=True
                    sc = 0


    pri_snap = np.asarray(pri_snap,dtype='int32')
    pri_sfid = np.asarray(pri_sfid,dtype='int64')
    pri_iz = np.asarray(pri_iz,dtype='float32')
    pri_tz = np.asarray(pri_tz,dtype='float32')
    pri_mstar = np.asarray(pri_mstar,dtype='float32')
    pri_mhalo = np.asarray(pri_mhalo,dtype='float32')
    pri_mbary = np.asarray(pri_mbary,dtype='float32')
    pri_nc = np.asarray(pri_nc,dtype='int32')

    sec_snap = np.asarray(sec_snap,dtype='int32')
    sec_sfid = np.asarray(sec_sfid,dtype='int64')
    sec_iz = np.asarray(sec_iz,dtype='float32')
    sec_tz = np.asarray(sec_tz,dtype='float32')
    sec_mstar = np.asarray(sec_mstar,dtype='float32')
    sec_mhalo = np.asarray(sec_mhalo,dtype='float32')
    sec_mbary = np.asarray(sec_mbary,dtype='float32')
    sec_drad_arcsec = np.asarray(sec_drad_arcsec,dtype='float32')

    print(pri_tz.shape, sec_mstar.shape)

    return pri_snap,pri_sfid,pri_iz,pri_tz,pri_mstar,pri_mhalo,pri_mbary,pri_nc,sec_snap,sec_sfid,sec_iz,sec_tz,sec_mstar,sec_mhalo,sec_mbary,sec_drad_arcsec


def evaluate_orbit(xx,yy,zz,vx,vy,vz,tz,snapid,sfid,pri_snap,pri_sfid,sec_snap,sec_sfid,pri_tz,sec_tz,pri_mass,sec_mass):

    pi = np.where(np.logical_and(np.logical_and(snapid==pri_snap,sfid==pri_sfid),np.abs(tz - pri_tz) < 0.001))[0]
    si = np.where(np.logical_and(np.logical_and(snapid==sec_snap,sfid==sec_sfid),np.abs(tz - sec_tz) < 0.001))[0]

    #print pi
    #print si

    assert pi.shape[0]==1
    assert si.shape[0]==1

    Mpc_in_m = 3.086e22
    G_m_kg_s = G.value  #m^3 / (kg s^2)
    Msun_in_kg = 1.99e30
    pri_mass_kg = pri_mass*Msun_in_kg
    sec_mass_kg = sec_mass*Msun_in_kg
    
    p_r = np.asarray([xx[pi],yy[pi],zz[pi]]).flatten()  #Mpc
    p_v = np.asarray([vx[pi],vy[pi],vz[pi]]).flatten()*1.0e3  #km/s -> m/s

    s_r = np.asarray([xx[si],yy[si],zz[si]]).flatten()
    s_v = np.asarray([vx[si],vy[si],vz[si]]).flatten()*1.0e3
   
    r_vec = (p_r - s_r)*Mpc_in_m
    v_vec = (p_v - s_v)

    r = np.linalg.norm(r_vec)
    r2 = (r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)**0.5

    rdot = np.linalg.norm(v_vec)
    mu = Msun_in_kg*pri_mass*sec_mass/(pri_mass + sec_mass)  #Msun -> kg


    kE = 0.5*mu*(rdot)**2
    pE = - G_m_kg_s*pri_mass_kg*sec_mass_kg/r2
    E = kE + pE

    L = np.linalg.norm(mu*np.cross(r_vec,v_vec))

    eccentricity = (1.0 + (2.0*E*L**2)/(mu*(G_m_kg_s*pri_mass_kg*sec_mass_kg)**2))**0.5
    rperi_kpc = (1.0e3*(L**2)/((1.0 + eccentricity)*mu*G_m_kg_s*pri_mass_kg*sec_mass_kg))/Mpc_in_m #m-->Kpc
    b_kpc = (1.0e3)*(L/(mu*(2.0*E/mu)**0.5))/Mpc_in_m

    if eccentricity < 1.0:
        rapo_kpc = (1.0e3*(L**2)/((1.0 - eccentricity)*mu*G_m_kg_s*pri_mass_kg*sec_mass_kg))/Mpc_in_m #m-->Kpc
    else:
        rapo_kpc = 1.0e3*r2/Mpc_in_m

    print(' ')
    print(r, r2, kE, pE, E, eccentricity, rperi_kpc)
    print(' ')


    return eccentricity, rperi_kpc, 1.0e3*r2/Mpc_in_m, rdot/1.0e3, rapo_kpc, b_kpc



def evaluate_environment(bp,pri_snap,pri_sfid,sec_snap,sec_sfid):

    efile = os.path.join(bp,'vrg_environment/environment_{:03d}'.format(pri_snap)+'.hdf5')
    delta = None
    hsml = None

    if not os.path.lexists(efile):
        return -1

    with h5py.File(efile,'r') as f:
        delta = f['delta'][pri_sfid]
        #hsml = f['hsml'][pri_sfid]

    assert delta is not None
 
    return delta #, hsml


def evaluate_fluxes(bp,pri_snap,pri_sfid,sec_snap,sec_sfid,gmag_all,snapid,sfid,tz,pri_tz,sec_tz):
    pi = np.where(np.logical_and(np.logical_and(snapid==pri_snap,sfid==pri_sfid),np.abs(tz - pri_tz) < 0.001))[0]
    si = np.where(np.logical_and(np.logical_and(snapid==sec_snap,sfid==sec_sfid),np.abs(tz - sec_tz) < 0.001))[0]
    assert pi.shape[0]==1
    assert si.shape[0]==1
    
    pri_gmag = np.asarray(gmag_all[pi])[0]
    sec_gmag = np.asarray(gmag_all[si])[0]

    gmag_ratio = 10.0**(0.4*(pri_gmag-sec_gmag))   #this yields f_sec/f_pri
    
    return pri_gmag, sec_gmag, gmag_ratio


def find_pairs(lightconefile,pairfile,sep=100.0,hh=0.704,massmin=10**(10.5),ratio=10.0,photz_sig=0.02,usemstar=True,bp = '/astro/snyder_lab2/Illustris/Illustris-2',label='mstar10.5'):

    pri_snap,pri_sfid,pri_iz,pri_tz,pri_mstar,pri_mhalo,pri_mbary,pri_nc,sec_snap,sec_sfid,sec_iz,sec_tz,sec_mstar,sec_mhalo,sec_mbary,sec_drad_arcsec = parse_pair_catalog(pairfile)

    data = ascii.read(lightconefile)
    outfile = os.path.basename(lightconefile)[0:11] + '_'+os.path.basename(lightconefile)[-15:-4]+'_'+label+'_pairs.txt'
    outfile_pri = os.path.basename(lightconefile)[0:11] + '_'+os.path.basename(lightconefile)[-15:-4]+'_'+label+'_primaries.txt'

    print(lightconefile)
    print(outfile)

    snapid=np.asarray(np.int32(data['col1']))
    shid = np.asarray(np.int32(data['col2']))
    ra = np.float64(data['col3'])
    dec = np.float64(data['col4'])
    tz = np.float64(data['col9'])
    iz = np.float64(data['col10'])
    scale = np.float32(data['col12'])
    xx = np.float32(data['col28'])  #proper Mpc
    yy = np.float32(data['col29'])
    zz = np.float32(data['col30'])
    lid = np.int32(data['col20'])
    mstar = np.float32(data['col21'])
    mgas = np.float32(data['col22'])
    mhalo = np.float32(data['col23'])
    mbary = np.float32(data['col25'])
    sfr = np.float32(data['col26'])
    gmag = np.float32(data['col31'])

    
    zmin = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]
    zmax = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    zmed = [0.1, 0.3, 0.5, 0.7, 0.9, 1.25,  1.75, 2.25, 2.75]

    vx = np.float32(data['col35'])
    vy = np.float32(data['col36'])
    vz = np.float32(data['col37'])

    separation = sep/hh

    if usemstar==True:
        primary = np.where(mstar >= massmin)[0]
        pairs = np.where(pri_mstar >= massmin)[0]
    else:
        primary = np.where(mhalo >= massmin)[0]
        pairs = np.where(pri_mhalo >= massmin)[0]

    print(primary.shape)

    #save a primary catalog
    with open(outfile_pri,'w') as outf:

        outf.write('#iz snap sfid mstar mhalo mbary delta hsml sfr\n')

        if primary.shape[0] > 0:
            for pi in primary:
                dz = photz_sig * (1.0 + iz[pi])



                #secondary = np.where(np.logical_and(np.logical_and(np.logical_and(iz >= iz[pi]-dz,iz <= iz[pi] + dz),mstar < mstar[pi]),mstar > (10.0**9.5)))[0]


                #rad = (separation/scale[pi])/3600.0 #radius cut, in degrees
                #ra_p = ra[pi]
                #dec_p = dec[pi]
                iz_p = iz[pi]
                #coord_p = SkyCoord(ra_p,dec_p,unit='deg')

                print(iz_p, pi)
                delta = evaluate_environment(bp,snapid[pi],shid[pi],snapid[pi],shid[pi])
                hsml = -1.0

                wl = '{:16.10f}  {:12d}  {:12d}  {:12.4e}  {:12.4e}  {:12.4e}  {:12.6f}  {:12.6f}  {:12.6f}'.format(iz_p,snapid[pi],shid[pi],mstar[pi],mhalo[pi],mbary[pi],delta,hsml,sfr[pi])
                outf.write(wl+'\n')


    #save pairs catalog
    with open(outfile,'w') as outf:
        outf.write('#iz snap sfid mstar mhalo mbary Nc sec_iz sec_snap sec_sfid sec_mstar sec_mhalo sec_mbary sec_drad_arcsec drad_kpc pairMerges mergersnap mergerage time_until_merger mstar_ratio_now mstar_ratio_tmax mhalo_ratio_tmax mhalo_ratio_now snap_tmax merger_mass merger_mstar ecc rperi delta hsml rnow vnow rapo b pri_sfr sec_sfr delta_z delta_v gratio bratio\n')
        for j,psnap in enumerate(pri_snap[pairs]):
            i = pairs[j]

            print(pri_snap[i],pri_sfid[i],sec_snap[i],sec_sfid[i])
            pairMerges, merger_snapshot, time_until_merger, time_of_merger, merger_redshift, merger_mass, merger_mstar, mass_ratio_tmax,mass_ratio_now,sec_snap_tmax,mhalo_ratio_tmax,pri_sfr,sec_sfr = evaluate_pair(bp,pri_snap[i],pri_sfid[i],sec_snap[i],sec_sfid[i])
            print(i, pri_tz[i], pri_iz[i], pairMerges, psnap, merger_snapshot, time_until_merger, mass_ratio_now, mass_ratio_tmax, sec_snap_tmax)

            ecc,rperi,rnow,vnow,rapo,b = evaluate_orbit(xx,yy,zz,vx,vy,vz,tz,snapid,shid,pri_snap[i],pri_sfid[i],sec_snap[i],sec_sfid[i],pri_tz[i],sec_tz[i],pri_mhalo[i],sec_mhalo[i])

            delta = evaluate_environment(bp,pri_snap[i],pri_sfid[i],sec_snap[i],sec_sfid[i])
            hsml = -1.0
            print(ecc, rperi, delta, hsml)

            sec_drad_kpc = np.float32( sec_drad_arcsec[i]/illcos.arcsec_per_kpc_proper(pri_iz[i]) )
            mhalo_ratio_now = sec_mhalo[i]/pri_mhalo[i]

            delta_z = np.abs(pri_iz[i]-sec_iz[i])
            delta_v = delta_z*3.0e5
            

            pri_g, sec_g, gratio = evaluate_fluxes(bp,pri_snap[i],pri_sfid[i],sec_snap[i],sec_sfid[i],gmag,snapid,shid,tz,pri_tz[i],sec_tz[i])
            bratio = sec_mbary[i]/pri_mbary[i]
            print(gratio, bratio)
            
            wl = '{:16.10f}  {:8d}  {:12d}  {:12.4e}  {:12.4e}  {:12.4e}  {:6d}'\
                 '  {:16.10f}  {:8d}  {:12d}  {:12.4e}  {:12.4e}  {:12.4e}  {:12.6f}  {:12.6f}'\
                 '  {:12s}  {:8d}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  {:8d}  {:12.4e}  {:12.4e}'\
                 '  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}  {:12.6f}'.format(pri_iz[i],pri_snap[i],pri_sfid[i],pri_mstar[i],pri_mhalo[i],pri_mbary[i],pri_nc[i],
                        sec_iz[i],sec_snap[i],sec_sfid[i],sec_mstar[i],sec_mhalo[i],sec_mbary[i],sec_drad_arcsec[i],sec_drad_kpc,
                        str(pairMerges),merger_snapshot,time_of_merger,time_until_merger,mass_ratio_now,mass_ratio_tmax,mhalo_ratio_tmax,mhalo_ratio_now,sec_snap_tmax,
                                                                                                                                                             merger_mass,merger_mstar,ecc,rperi,delta,hsml,rnow,vnow,rapo, b, pri_sfr, sec_sfr, delta_z, delta_v, gratio, bratio)

            outf.write(wl+'\n')
    return




if __name__=="__main__":
    #find_pairs('Illustris-2_RADEC_hudfwide_75Mpc_7_6_xyz.txt','/user/lotz/illustris/Illustris2_pairs_r100_h0.70_mstar10.5_mr10_dz0.020_mock1.list',usemstar=True,label='mstar10.5')
    #find_pairs('Illustris-2_RADEC_hudfwide_75Mpc_7_6_yxz.txt','/user/lotz/illustris/Illustris2_pairs_r100_h0.70_mstar10.5_mr10_dz0.020_mock2.list',usemstar=True,label='mstar10.5')
    #find_pairs('Illustris-2_RADEC_hudfwide_75Mpc_7_6_zyx.txt','/user/lotz/illustris/Illustris2_pairs_r100_h0.70_mstar10.5_mr10_dz0.020_mock3.list',usemstar=True,label='mstar10.5')

    #find_pairs('Illustris-1_RADEC_hudfwide_75Mpc_7_6_xyz_corners.txt','/astro/snyder_lab2/Illustris/Lightcones/forgreg/Illustris1_pairs_r100_h0.70_mstar10.5_mr10_dz0.020_mock1.list',
    #           usemstar=True,label='mstar10.5',bp = '/astro/snyder_lab2/Illustris/Illustris-1')
    #find_pairs('Illustris-1_RADEC_hudfwide_75Mpc_7_6_yxz_corners.txt','/astro/snyder_lab2/Illustris/Lightcones/forgreg/Illustris1_pairs_r100_h0.70_mstar10.5_mr10_dz0.020_mock2.list',
    #           usemstar=True,label='mstar10.5',bp = '/astro/snyder_lab2/Illustris/Illustris-1')
    #find_pairs('Illustris-1_RADEC_hudfwide_75Mpc_7_6_zyx_corners.txt','/astro/snyder_lab2/Illustris/Lightcones/forgreg/Illustris1_pairs_r100_h0.70_mstar10.5_mr10_dz0.020_mock3.list',
    #           usemstar=True,label='mstar10.5',bp = '/astro/snyder_lab2/Illustris/Illustris-1')

    #find_pairs('Illustris-1_RADEC_hudfwide_75Mpc_7_6_xyz_corners.txt','/astro/snyder_lab2/Illustris/Lightcones/new3/Illustris1_pairs_r100_h0.70_mstar10.5_mr100000_dz0.020_z10_mock1.list',
    #           usemstar=True,label='mstar10.5new3',bp = '/astro/snyder_lab2/Illustris/Illustris-1')
    #find_pairs('Illustris-1_RADEC_hudfwide_75Mpc_7_6_yxz_corners.txt','/astro/snyder_lab2/Illustris/Lightcones/new3/Illustris1_pairs_r100_h0.70_mstar10.5_mr100000_dz0.020_z10_mock2.list',
    #           usemstar=True,label='mstar10.5new3',bp = '/astro/snyder_lab2/Illustris/Illustris-1')
    #find_pairs('Illustris-1_RADEC_hudfwide_75Mpc_7_6_zyx_corners.txt','/astro/snyder_lab2/Illustris/Lightcones/new3/Illustris1_pairs_r100_h0.70_mstar10.5_mr100000_dz0.020_z10_mock3.list',
    #           usemstar=True,label='mstar10.5new3',bp = '/astro/snyder_lab2/Illustris/Illustris-1')

    #find_pairs('Illustris-1_RADEC_JWST1_75Mpc_7_6_xyz.txt','Pairs_7_6_xyz.txt',
    #           usemstar=True,label='mstar9.5',massmin=10.0**(9.5),bp = '/astro/snyder_lab2/Illustris/Illustris-1' )

    #find_pairs('Illustris-1_RADEC_JWST1_75Mpc_7_6_yxz.txt','Pairs_7_6_yxz.txt',
    #           usemstar=True,label='mstar9.5',massmin=10.0**(9.5),bp = '/astro/snyder_lab2/Illustris/Illustris-1' )

    find_pairs('Illustris-1_RADEC_JWST1_75Mpc_7_6_zyx.txt','Pairs_7_6_zyx.txt',
               usemstar=True,label='mstar9.5',massmin=10.0**(9.5),bp = '/astro/snyder_lab2/Illustris/Illustris-1' )

    #pass

    #find_pairs('Illustris-2_RADEC_hudfwide_75Mpc_7_6_xyz.txt','/user/lotz/illustris/Illustris2_pairs_r100_h0.70_mhalo10.5_mr10_dz0.020_mock1.list',usemstar=False,label='mhalo11.5',massmin=10.0**(11.5))
    #find_pairs('Illustris-2_RADEC_hudfwide_75Mpc_7_6_yxz.txt','/user/lotz/illustris/Illustris2_pairs_r100_h0.70_mhalo10.5_mr10_dz0.020_mock2.list',usemstar=False,label='mhalo11.5',massmin=10.0**(11.5))
    #find_pairs('Illustris-2_RADEC_hudfwide_75Mpc_7_6_zyx.txt','/user/lotz/illustris/Illustris2_pairs_r100_h0.70_mhalo10.5_mr10_dz0.020_mock3.list',usemstar=False,label='mhalo11.5',massmin=10.0**(11.5))
