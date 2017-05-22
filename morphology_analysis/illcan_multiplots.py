import math
import string
import sys
import struct
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.colors as pycolors
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import numpy.ma as ma
import asciitable
import scipy.ndimage
import scipy.stats as ss
import scipy.signal
import scipy as sp
import scipy.odr as odr
from scipy.integrate import quad
import glob
import os
import gzip
import shutil
import astropy.io.ascii as ascii
import warnings
import subprocess
import astropy
import astropy.io.fits as pyfits
import astropy.units as u
from astropy.cosmology import WMAP7,z_at_value
import copy
import datetime
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
import PyML
import PyML.machinelearning
import gfs_twod_histograms as gth
import pandas
import h5py
import make_color_image
import photutils
import gfs_sublink_utils as gsu
import random
import showgalaxy
import masshistory



ilh = 0.704
illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)
sq_arcsec_per_sr = 42545170296.0


all_snap_keys = ['snapshot_103','snapshot_085','snapshot_075','snapshot_068',
                 'snapshot_064','snapshot_060','snapshot_054', 'snapshot_049',
                 'snapshot_045','snapshot_041','snapshot_038','snapshot_035']

all_fil_keys =  ['WFC3-F336W', 'ACS-F435W', 'ACS-F606W', 'ACS-F814W',  'WFC3-F105W', 'WFC3-F160W',
                 'NC-F115W',  'NC-F150W', 'NC-F200W',   'NC-F277W', 'NC-F356W', 'NC-F444W']    

im_snap_keys = ['snapshot_103','snapshot_085','snapshot_075','snapshot_068',
             'snapshot_064','snapshot_060','snapshot_054', 'snapshot_049']

im_rf_fil_keys = ['ACS-F606W','ACS-F814W','WFC3-F105W','NC-F150W',
                  'NC-F150W','NC-F200W','NC-F200W','NC-F277W']

im_rf_fil_keys_hst= ['ACS-F606W','ACS-F814W','WFC3-F105W','WFC3-F160W',
                  'WFC3-F160W','WFC3-F160W','WFC3-F160W','WFC3-F160W']


snap_keys = im_snap_keys
fil_keys = im_rf_fil_keys
fil_keys_hst=im_rf_fil_keys_hst


data_z1_keys=[0.25,0.75,1.25,1.75,2.25,2.75,3.5,4.5]
data_z2_keys=[0.75,1.25,1.75,2.25,2.75,3.5,4.5,5.5]



im_npix = [600,450,300,300,
           200,200,100,100]

im_fil_keys = {}
im_fil_keys['snapshot_103']={}
im_fil_keys['snapshot_103']['b']=['NC-F115W']
im_fil_keys['snapshot_103']['g']=['NC-F150W']
im_fil_keys['snapshot_103']['r']=['NC-F200W']

im_fil_keys['snapshot_085']={}
im_fil_keys['snapshot_085']['b']=['NC-F115W']
im_fil_keys['snapshot_085']['g']=['NC-F150W']
im_fil_keys['snapshot_085']['r']=['NC-F200W']

im_fil_keys['snapshot_075']={}
im_fil_keys['snapshot_075']['b']=['NC-F115W']
im_fil_keys['snapshot_075']['g']=['NC-F150W']
im_fil_keys['snapshot_075']['r']=['NC-F200W']

im_fil_keys['snapshot_068']={}
im_fil_keys['snapshot_068']['b']=['NC-F115W']
im_fil_keys['snapshot_068']['g']=['NC-F150W']
im_fil_keys['snapshot_068']['r']=['NC-F200W']

im_fil_keys['snapshot_064']={}
im_fil_keys['snapshot_064']['b']=['NC-F115W']
im_fil_keys['snapshot_064']['g']=['NC-F150W']
im_fil_keys['snapshot_064']['r']=['NC-F200W']

im_fil_keys['snapshot_060']={}
im_fil_keys['snapshot_060']['b']=['NC-F115W']
im_fil_keys['snapshot_060']['g']=['NC-F150W']
im_fil_keys['snapshot_060']['r']=['NC-F200W']

im_fil_keys['snapshot_054']={}
im_fil_keys['snapshot_054']['b']=['NC-F115W']
im_fil_keys['snapshot_054']['g']=['NC-F150W']
im_fil_keys['snapshot_054']['r']=['NC-F200W']

im_fil_keys['snapshot_049']={}
im_fil_keys['snapshot_049']['b']=['NC-F115W']
im_fil_keys['snapshot_049']['g']=['NC-F150W']
im_fil_keys['snapshot_049']['r']=['NC-F200W']

#            cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc']

def load_candels_dfs(field='egs',zrange=[1.75,2.25],mrange=[10.50,13.5],col_labels=['dGM20','fGM20','ASYM','MPRIME','I','D','CON']):
    if field=='all':
        pass

    dirn='/Users/gsnyder/Dropbox/Workspace/Papers_towrite/Illustris_ObservingMergers/CATALOGS/CANDELS_LOTZ'
    
    sedf=os.path.join(dirn,  field+'_zbest_sedfit_oct15.fits')
    sedhdulist=pyfits.open(sedf)
    sedtab=sedhdulist[1].data
    z=sedtab['Z_BEST']
    logm=sedtab['LMSTAR_BC03']
    label=sedtab['CANDELS_ID']

    
    mfi=os.path.join(dirn, field+'_acs_f814w_30mas_feb17_morph.fits')
    mfj=os.path.join(dirn, field+'_wfc3_f125w_60mas_feb17_morph.fits')
    mfh=os.path.join(dirn, field+'_wfc3_f160w_60mas_feb17_morph.fits')

    mfitab=pyfits.open(mfi)[1].data
    mfjtab=pyfits.open(mfj)[1].data
    mfhtab=pyfits.open(mfh)[1].data

    mfi_id=mfitab['CANDELS_ID']
    mfj_id=mfjtab['CANDELS_ID']
    mfh_id=mfhtab['CANDELS_ID']

    mfi_mstar=np.zeros_like(mfi_id,dtype=np.float64)
    mfi_z=np.zeros_like(mfi_id,dtype=np.float64)
    
    mfj_mstar=np.zeros_like(mfj_id,dtype=np.float64)
    mfj_z=np.zeros_like(mfj_id,dtype=np.float64)

    mfh_mstar=np.zeros_like(mfh_id,dtype=np.float64)
    mfh_z=np.zeros_like(mfh_id,dtype=np.float64)
    
    
    for i,mfi_label in enumerate(mfi_id):
        ix= label==mfi_label
        mfi_mstar[i]=logm[ix]
        mfi_z[i]=z[ix]
    for i,mfj_label in enumerate(mfj_id):
        ix= label==mfj_label
        mfj_mstar[i]=logm[ix]
        mfj_z[i]=z[ix]
    for i,mfh_label in enumerate(mfh_id):
        ix= label==mfh_label
        mfh_mstar[i]=logm[ix]
        mfh_z[i]=z[ix]

    if col_labels[0]=='dGM20':
        mfi_g=mfitab['GINI_I']
        mfi_m=mfitab['M20_I']
        mfi_1=SGM20(mfi_g,mfi_m)
        mfi_2=FGM20(mfi_g,mfi_m)
        
        mfj_g=mfjtab['GINI_J']
        mfj_m=mfjtab['M20_J']
        mfj_1=SGM20(mfj_g,mfj_m)
        mfj_2=FGM20(mfj_g,mfj_m)

        mfh_g=mfhtab['GINI_H']
        mfh_m=mfhtab['M20_H']
        mfh_1=SGM20(mfh_g,mfh_m)
        mfh_2=FGM20(mfh_g,mfh_m)
    else:
        mfi_1=mfitab['GINI_I']
        mfi_2=mfitab['M20_I']
        mfj_1=mfjtab['GINI_J']
        mfj_2=mfjtab['M20_J']
        mfh_1=mfhtab['GINI_H']
        mfh_2=mfhtab['M20_H']

    iix=(mfi_mstar >= mrange[0])*(mfi_mstar < mrange[1])*(mfi_z >= zrange[0])*(mfi_z < zrange[1])
    jix=(mfj_mstar >= mrange[0])*(mfj_mstar < mrange[1])*(mfj_z >= zrange[0])*(mfj_z < zrange[1])
    hix=(mfh_mstar >= mrange[0])*(mfh_mstar < mrange[1])*(mfh_z >= zrange[0])*(mfh_z < zrange[1])

    
    dict_814={}
    dict_814[col_labels[0]]=mfi_1[iix]
    dict_814[col_labels[1]]=mfi_2[iix]
    dict_814[col_labels[2]]=mfitab[col_labels[2]+'_I'][iix]
    dict_814[col_labels[3]]=mfitab[col_labels[3]+'_I'][iix]
    dict_814[col_labels[4]]=mfitab[col_labels[4]+'_I'][iix]
    dict_814[col_labels[5]]=mfitab[col_labels[5]+'_I'][iix]
    try:
        dict_814[col_labels[6]]=mfitab[col_labels[6]+'_I'][iix]
    except:
        dict_814[col_labels[6]]=mfitab['C_I'][iix]

    dict_814['LMSTAR_BC03']=mfi_mstar[iix]
    dict_814['Z_BEST']=mfi_z[iix]
    dict_814['CANDELS_ID']=mfi_id[iix]
    
        
    dict_125={}
    dict_125[col_labels[0]]=mfj_1[jix]
    dict_125[col_labels[1]]=mfj_2[jix]
    dict_125[col_labels[2]]=mfjtab[col_labels[2]+'_J'][jix]
    dict_125[col_labels[3]]=mfjtab[col_labels[3]+'_J'][jix]
    dict_125[col_labels[4]]=mfjtab[col_labels[4]+'_J'][jix]
    dict_125[col_labels[5]]=mfjtab[col_labels[5]+'_J'][jix]
    dict_125[col_labels[6]]=mfjtab[col_labels[6]+'_J'][jix]

    dict_125['LMSTAR_BC03']=mfj_mstar[jix]
    dict_125['Z_BEST']=mfj_z[jix]
    dict_125['CANDELS_ID']=mfj_id[jix]
    
    dict_160={}
    dict_160[col_labels[0]]=mfh_1[hix]
    dict_160[col_labels[1]]=mfh_2[hix]
    dict_160[col_labels[2]]=mfhtab[col_labels[2]+'_H'][hix]
    dict_160[col_labels[3]]=mfhtab[col_labels[3]+'_H'][hix]
    dict_160[col_labels[4]]=mfhtab[col_labels[4]+'_H'][hix]
    dict_160[col_labels[5]]=mfhtab[col_labels[5]+'_H'][hix]
    dict_160[col_labels[6]]=mfhtab[col_labels[6]+'_H'][hix]

    dict_160['LMSTAR_BC03']=mfh_mstar[hix]
    dict_160['Z_BEST']=mfh_z[hix]
    dict_160['CANDELS_ID']=mfh_id[hix]
    
    df_814 = pandas.DataFrame(dict_814)
    df_125 = pandas.DataFrame(dict_125)
    df_160 = pandas.DataFrame(dict_160)
    
    
    return df_814,df_125,df_160


def VRG_dndt(mulo,muhi,M,z):

    dndt,abserr = quad(VRG_dndmudt,mulo,muhi,args=(M,z))

    return dndt,abserr


def VRG_dndmudt(mu,M,z):
    M0 = 2.0e11
    log10A = -2.2287
    eta = 2.4644
    alpha0 = 0.2241
    alpha1 = -1.1759
    beta0 = -1.2595
    beta1 = 0.0611
    gamma = -0.0477
    delta0 = 0.7668
    delta1 = -0.4695

    Az = ((1.0 + z)**eta)*10.0**(log10A) #Gyr^-1
    alphaz = alpha0*(1.0 + z)**(alpha1)
    betaz = beta0*(1.0 + z)**(beta1)
    deltaz = delta0*(1.0 + z)**(delta1)

    dndmudt = Az*((M/1.0e10)**alphaz)*(1.0 + (M/M0)**(deltaz))*(mu**(betaz + gamma*np.log10(M/1.0e10)))

    return dndmudt


def return_rf_variables():
    return np.asarray(im_snap_keys), np.asarray(im_rf_fil_keys), np.asarray(im_npix)


def simple_classifier_stats(classifications,labels):

    tp = np.where( (labels==True)*(classifications==True) )[0]
    fn = np.where( (labels==True)*(classifications==False))[0]
    fp = np.where( (labels==False)*(classifications==True))[0]

    completeness = float(tp.shape[0])/float(tp.shape[0] + fn.shape[0])
    ppv = float(tp.shape[0])/float(tp.shape[0] + fp.shape[0])
    
    return completeness, ppv

def FGM20(gini,m20):
    a = -0.14
    b = 1.0
    c = -0.80

    gini_f = gini*1.0
    m20_f = m20*1.0

    d = np.abs(a*m20_f + b*gini_f + c)/((a**2 + b**2)**0.5)
    si = np.where(gini_f < -1.0*(a/b)*m20_f - (c/b))[0]


    d[si] = -1.0*d[si]

    f = d*1.0


    return f


def SGM20(gini,m20):

    a = 0.14 ; b = 1.0 ; c = -0.33
    gini_f = gini*1.0
    m20_f = m20*1.0

    d = np.abs(a*m20_f + b*gini_f + c)/((a**2 + b**2)**0.5)
    si = np.where(gini_f < -1.0*(a/b)*m20_f - (c/b))

    d[si] = -1.0*d[si]

    f = d*1.0

    return f


def get_all_morph_val(msF,sk,fk,keyname,camera=None):
    if camera is None:
        morph_array = np.concatenate( (msF['nonparmorphs'][sk][fk]['CAMERA0'][keyname].value,
                                       msF['nonparmorphs'][sk][fk]['CAMERA1'][keyname].value,
                                       msF['nonparmorphs'][sk][fk]['CAMERA2'][keyname].value,
                                       msF['nonparmorphs'][sk][fk]['CAMERA3'][keyname].value) )
    else:
        morph_array = msF['nonparmorphs'][sk][fk][camera][keyname].value

    return morph_array


def get_all_snap_val(msF,sk,keyname,camera=None):
    if camera is None:
        val_array = np.concatenate( (msF['nonparmorphs'][sk][keyname].value,
                                     msF['nonparmorphs'][sk][keyname].value,
                                     msF['nonparmorphs'][sk][keyname].value,
                                     msF['nonparmorphs'][sk][keyname].value) )
    else:
        val_array = msF['nonparmorphs'][sk][keyname].value

    return val_array


def get_mergerinfo_val(merF,sk,keyname):
    val_array = np.concatenate( (merF['mergerinfo'][sk][keyname].value,
                                   merF['mergerinfo'][sk][keyname].value,
                                   merF['mergerinfo'][sk][keyname].value,
                                   merF['mergerinfo'][sk][keyname].value) )
    return val_array



def label_merger1(merF,sk):
    latest_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLastGyr')
    label_boolean = latest_NumMajorMergersLastGyr >= 1.0
    return label_boolean


#minor now really means minor
def label_merger4(merF,sk):
    latest_NumMinorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLastGyr')
    latest_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLastGyr')
    
    label_boolean = np.logical_or( latest_NumMinorMergersLastGyr >= 1.0, latest_NumMajorMergersLastGyr >= 1.0)
    
    return label_boolean


def label_merger_window500_major(merF,sk):
    latest_NumMajorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    label_boolean = latest_NumMajorMergerLast500Myr >= 1.0
    return label_boolean    


#assumes merger_span = 0.25 Gyr so this covers 500Myr around image
def label_merger_window500_both(merF,sk):
    latest_NumMajorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    latest_NumMinorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast500Myr')
    merger_number = latest_NumMajorMergerLast500Myr + latest_NumMinorMergerLast500Myr

    label_boolean = np.logical_or( latest_NumMajorMergerLast500Myr >= 1.0, latest_NumMinorMergerLast500Myr >= 1.0 )
    print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number

#assumes merger_span = 0.25 Gyr so this covers 250Myr after image
def label_merger_forward250_both(merF,sk):
    latest_NumMajorMergerLast250Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast250Myr')
    latest_NumMinorMergerLast250Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast250Myr')
    merger_number = latest_NumMajorMergerLast250Myr + latest_NumMinorMergerLast250Myr

    label_boolean = np.logical_or( latest_NumMajorMergerLast250Myr >= 1.0, latest_NumMinorMergerLast250Myr >= 1.0 )
    print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number


def label_merger_past250_both(merF,sk):
    this_NumMajorMergerLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast250Myr')
    this_NumMinorMergerLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast250Myr')
    merger_number = this_NumMajorMergerLast250Myr + this_NumMinorMergerLast250Myr

    label_boolean = np.logical_or( this_NumMajorMergerLast250Myr >= 1.0, this_NumMinorMergerLast250Myr >= 1.0 )
    print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number


def label_merger_past500_both(merF,sk):
    this_NumMajorMergerLast500Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast500Myr')
    this_NumMinorMergerLast500Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast500Myr')
    merger_number = this_NumMajorMergerLast500Myr + this_NumMinorMergerLast500Myr

    label_boolean = np.logical_or( this_NumMajorMergerLast500Myr >= 1.0, this_NumMinorMergerLast500Myr >= 1.0 )
    print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number


#!!!!!
#NOTE USE ONLY IF MERGERDATA FILE HAS SPAN=0.5 Gyr i.e. the March2017 numbers not the May2017 one!!!
#assumes merger_span = 0.5 Gyr so this covers 500Myr after image
def label_merger_forward500_both(merF,sk):
    latest_NumMajorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    latest_NumMinorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast500Myr')
    merger_number = latest_NumMajorMergerLast500Myr + latest_NumMinorMergerLast500Myr

    label_boolean = np.logical_or( latest_NumMajorMergerLast500Myr >= 1.0, latest_NumMinorMergerLast500Myr >= 1.0 )
    print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number



def make_pc_dict(msF,sk,fk):

    parameters = ['C','M20','GINI','ASYM','MPRIME','I','D']
    pcd = {}
    pcd['C'] = get_all_morph_val(msF,sk,fk,'CC')
    pcd['M20'] = get_all_morph_val(msF,sk,fk,'M20')
    pcd['GINI'] = get_all_morph_val(msF,sk,fk,'GINI')
    pcd['ASYM'] = get_all_morph_val(msF,sk,fk,'ASYM')
    pcd['MPRIME'] = get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
    pcd['I'] = get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
    pcd['D'] = get_all_morph_val(msF,sk,fk,'MID1_DSTAT')

    npmorph = PyML.machinelearning.dataMatrix(pcd,parameters)
    pc = PyML.machinelearning.pcV(npmorph)

    return parameters, pcd, pc, pcd



def plot_sfr_radius_mass(msF,merF,sk,fk,FIG,xlim,ylim,rlim,Cval,gridf='median_grid',**bin_kwargs):

    sfr  = get_all_snap_val(msF,sk,'SFR_Msunperyr')
    mstar  = get_all_snap_val(msF,sk,'Mstar_Msun')
    
    
    redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
    
    
    labs=11
    axi = FIG.add_subplot(2,1,1)
    axi.set_ylabel('$log_{10}$ SFR [$M_{\odot} yr^{-1}$]',labelpad=1,size=labs)    
    axi.tick_params(axis='both',which='both',labelsize=labs)
    axi.locator_params(nbins=5,prune='both')

    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])

    #if given array, use it
    if type(Cval)==np.ndarray:
        gi = np.where(np.isfinite(Cval)==True)[0]
        data_dict = {'x':Cval[gi]}
    else:
        #otherwise, loop over dict keys and send to plotter
        #this assumes that all values are finite!
        gi = np.arange(mstar.shape[0])
        data_dict = copy.copy(Cval)

    
    res = gth.make_twod_grid(axi,np.log10(mstar[gi]),np.log10(sfr[gi]),data_dict,gridf,**bin_kwargs)
    

    axi = FIG.add_subplot(2,1,2)
    axi.set_ylabel('$log_{10}\ R_p$ [$kpc$]',labelpad=1,size=labs)
    axi.set_xlabel('$log_{10}\ M_{*} [M_{\odot}]$',labelpad=1,size=labs)
    axi.tick_params(axis='both',which='both',labelsize=labs)
    axi.locator_params(nbins=5,prune='both')
    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(rlim[0],rlim[1])

    
    kpc_per_arcsec = illcos.kpc_proper_per_arcmin(redshift).value/60.0

    size_pixels = get_all_morph_val(msF,sk,fk,'RP')
    pix_arcsec  = get_all_morph_val(msF,sk,fk,'PIX_ARCSEC')

    size_kpc = size_pixels*pix_arcsec*kpc_per_arcsec

    
    axi,colorobj = gth.make_twod_grid(axi,np.log10(mstar[gi]),np.log10(size_kpc[gi]),data_dict,gridf,**bin_kwargs)
    axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,0.05),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    axi.annotate(fk                             ,xy=(0.80,0.15),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)

    return colorobj



def make_rf_evolution_plots(snap_keys,fil_keys,rflabel='paramsmod',labelfunc='label_merger1',rf_masscut=0.0,twin=0.5):


    plot_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_global_stats.pdf'
    if not os.path.lexists('rf_plots'):
        os.mkdir('rf_plots')
    if not os.path.lexists('rf_plots/'+labelfunc):
        os.mkdir('rf_plots/'+labelfunc)

        
    f1 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)


    imp_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_importance_stats.pdf'
    f2 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)



    fm_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_merger_stats.pdf'
    f5 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.22, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)


    


    
    labs=11
    axi = f1.add_subplot(1,1,1)
    axi.set_xlim(0.2,4.5)
    axi.set_ylim(-0.01,1.1)
    axi.tick_params(axis='both',which='both',labelsize=labs)
    axi.locator_params(nbins=7,prune='both')

    axi.set_xlabel('redshift')
    axi.set_ylabel('value')
    
    axi2 = f2.add_subplot(1,1,1)
    axi2.set_xlim(0.2,4.5)
    axi2.set_ylim(0.3,1.99)
    axi2.tick_params(axis='both',which='both',labelsize=labs)
    axi2.locator_params(nbins=7,prune='both')

    axi2.set_xlabel('redshift')
    axi2.set_ylabel('relative importance')
  

    axi5 = f5.add_subplot(1,1,1)
    axi5.set_xlim(0.2,4.5)
    axi5.set_ylim(1.0e-3,2.0)
    axi5.tick_params(axis='both',which='both',labelsize=labs)
    axi5.locator_params(nbins=7,prune='both')

    axi5.set_xlabel(r'$redshift$',size=labs)
    axi5.set_ylabel(r'$fraction$',size=labs)

    #fakez=np.linspace(0.5,5.0,50)
    #ratez=np.zeros_like(fakez)

    #for i,z in enumerate(fakez):
    #    vrg_rate,vrg_err=VRG_dndt(0.10,1.0,10.0**10.5,z)
    #    ratez[i]=vrg_rate
  
    #axi5.semilogy(fakez,ratez*0.5,linestyle='solid',color='black',linewidth=1.5)

    vrg_z=[]
    vrg_rate=[]

    imp_norm = 1.0/7.0

    if rflabel=='paramsmod':
        cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc']
        syms=['^g','<r','ok','sb','sg','sy','*k']
        
    elif rflabel=='params':
        cols=['gini','m20','asym','Mstat','Istat','Dstat','cc']
        syms=['^g','<r','ok','sb','sg','sy','*k']
    else:   
        assert(False)


    #orig_snap_keys = snap_keys
    #orig_fil_keys = fil_keys

    snap_keys.append('combined')
    fil_keys.append('subset')
    
    for sk,fk in zip(snap_keys,fil_keys):
        if sk != 'combined':
            parameters, pcd, pc, pcd = make_pc_dict(msF,sk,fk)
            pc1 = pc.X[:,0].flatten()
            pc2 = pc.X[:,1].flatten()
            pc3 = pc.X[:,2].flatten()
            pc4 = pc.X[:,3].flatten()
            pc5 = pc.X[:,4].flatten()
            pc6 = pc.X[:,5].flatten()
            pc7 = pc.X[:,6].flatten()
            PCs=pandas.DataFrame(pc.X)
        
            asym = get_all_morph_val(msF,sk,fk,'ASYM')
            gini = get_all_morph_val(msF,sk,fk,'GINI')
            m20 = get_all_morph_val(msF,sk,fk,'M20')
            cc = get_all_morph_val(msF,sk,fk,'CC')
            Mstat = get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
            Istat = get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
            Dstat = get_all_morph_val(msF,sk,fk,'MID1_DSTAT')

            #S_GM20 = SGM20(gini,m20)
            #F_GM20 = FGM20(gini,m20)
        
        
            latest_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLastGyr')
            boolean_merger1 = latest_NumMajorMergersLastGyr >= 1.0
            latest_NumMinorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLastGyr')
            boolean_merger4 = latest_NumMinorMergersLastGyr >= 1.0
            
            this_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLastGyr')
            boolean_merger2 = this_NumMajorMergersLastGyr >= 1.0
        

            mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
            mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
            log_mstar_mhalo = np.log10( mstar/mhalo )

            redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
            ins = fk.split('-')[0]
            if redshift > 4.2:
                continue
        

        rfdata = 'rfoutput/'+labelfunc+'/'
        
        data_file = rfdata+rflabel+'_traindata_{}_{}.pkl'.format(sk,fk)
        result_file= rfdata+rflabel+'_result_{}_{}.pkl'.format(sk,fk)
        labels_file = rfdata+rflabel+'_labels_{}_{}.pkl'.format(sk,fk)
        prob_file = rfdata+rflabel+'_label_probability_{}_{}.pkl'.format(sk,fk)
        pcs_file= rfdata+rflabel+'_pc_{}_{}.pkl'.format(sk,fk)
        roc_file= rfdata+rflabel+'_roc_{}_{}.npy'.format(sk,fk)


        
        rf_data = np.load(data_file,encoding='bytes')
        rf_asym = rf_data['asym'].values
        rf_flag = rf_data['mergerFlag'].values
        rf_dgm20 = rf_data['dGM20'].values
        rf_cc = rf_data['cc'].values
        rf_mstar = rf_data['Mstar_Msun'].values
        rf_number = rf_data['mergerNumber'].values

        mergers_per_true=np.sum(rf_number)/np.sum(rf_flag)

        asym_classifier = rf_asym > 0.25
        asym_com,asym_ppv = simple_classifier_stats(asym_classifier,rf_flag)

        dgm_classifier = rf_dgm20 > 0.10
        dgm_com,dgm_ppv = simple_classifier_stats(dgm_classifier,rf_flag)

        #three_classifier = np.logical_or(rf_asym > 0.35,rf_sgm20 > 0.10)
        #three_com,three_ppv = simple_classifier_stats(three_classifier,rf_flag)
        

        result = np.load(result_file)
        completeness = np.median(result['completeness'].values)
        ppv = np.median(result['ppv'].values)
        probs = np.load(prob_file)
        iters = result['completeness'].values.shape[0]

        rfprob=np.asarray( np.mean(probs[[0,1,2,3,4]],axis=1))

        rf_class = rfprob > 0.4


        if sk !='combined':
            print('redshift: {:3.1f}   filter: {:15s}   RF sample size: {:8d}   # True mergers: {}   Avg Mstar: {}'.format(redshift,fk,rf_asym.shape[0],np.sum(rf_flag),np.median(rf_mstar)))
            vrg_z.append(redshift)
            vrg_rate_z,vrg_err_z=VRG_dndt(0.10,1.0,np.median(rf_mstar),redshift)

            vrg_rate.append(vrg_rate_z)

            for ck,sym in zip(cols,syms):
                imp = np.median(result[ck])/imp_norm
                axi2.plot(redshift,imp,sym,markersize=6)

            if ins=='WFC3':
                symbol='o'
                symcol='black'
                symsiz=8
            if ins=='ACS':
                symbol='s'
                symcol='Blue'
                symsiz=8
            if ins=='NC':
                symbol='^'
                symcol='red'
                symsiz=8
            
        
            axi.plot(redshift,completeness,marker=symbol,markersize=symsiz,markerfacecolor=symcol,markeredgecolor='None')
            axi.plot(redshift,ppv,marker=symbol,markersize=symsiz,markeredgecolor=symcol,markerfacecolor='None')

            axi.plot(redshift,asym_com,marker=symbol,markersize=symsiz/2,markerfacecolor=symcol,markeredgecolor='None')
            axi.plot(redshift,asym_ppv,marker=symbol,markersize=symsiz/2,markeredgecolor=symcol,markerfacecolor='None')


            axi5.semilogy(redshift,np.sum(asym_classifier)/asym_classifier.shape[0],'^g',markersize=symsiz/2.0)
            axi5.semilogy(redshift,np.sum(dgm_classifier)/dgm_classifier.shape[0],'^r',markersize=symsiz/2.0)

            #multiply by average N mergers in a "true" classification
            axi5.semilogy(redshift,mergers_per_true*np.sum(rf_class)/rf_class.shape[0],'sb',markersize=symsiz)
            
            axi5.semilogy(redshift,mergers_per_true*np.sum(rf_flag)/rf_flag.shape[0],'ok',markersize=symsiz)
            

            if sk=='snapshot_103':
                axi5.legend([ 'A > 0.25','dGM > 0.1', 'RF model (thresh=0.4)',r'$f_{merge}$'],loc='lower center',fontsize=10,numpoints=1)

        else:
            axi.plot([0,5],[completeness,completeness],linestyle='solid',marker='None')    
            axi.plot([0,5],[ppv,ppv],linestyle='dotted',marker='None')    


        roc_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_roc_stats.pdf'
        f3 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
        pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)
        axi3=f3.add_subplot(1,1,1)
        
        roc = (np.load(roc_file)).all()
        axi3.plot(roc['fpr'],roc['tpr'])
        axi3.set_xlim(-0.05,1.05)
        axi3.set_ylim(-0.05,1.05)
        axi3.set_xlabel('false positive rate')
        axi3.set_ylabel('true positive rate')
        f3.savefig(roc_filen,dpi=300)
        pyplot.close(f3)


        
        scat_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_scatterplot.pdf'
        f4 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
        pyplot.subplots_adjust(left=0.22, right=0.99, bottom=0.17, top=0.99,wspace=0.0,hspace=0.0)
    
        axi4=f4.add_subplot(1,1,1)
        axi4.locator_params(nbins=5,prune='both')

        axi4.semilogy(np.log10(rf_mstar),np.mean(probs[[0,1,2]],axis=1),'ok',markersize=4)
        axi4.semilogy(np.log10(rf_mstar[rf_flag==True]),np.mean(probs[[0,1,2]],axis=1)[rf_flag==True],'o',markersize=6,markerfacecolor=None,markeredgecolor='Orange')

        axi4.legend(['nonmerger','true merger'],loc='lower right',fontsize=10,framealpha=1.0)
        sp=12
        axi4.set_xlim(10.4,11.95)
        axi4.set_ylim(0.05,1.10)
        axi4.set_xlabel('log$_{10} M_*$',size=sp)
        axi4.set_ylabel('HST morphology merger probability',size=sp)
        axi4.tick_params(labelsize=sp)

        f4.savefig(scat_filen,dpi=300)
        pyplot.close(f4)
        
    f1.savefig(plot_filen,dpi=300)
    pyplot.close(f1)

    axi2.legend(cols,fontsize=11,ncol=3,loc='upper center',numpoints=1)
    f2.savefig(imp_filen,dpi=300)
    pyplot.close(f2)

    #500 Myr window implies predicted merger fraction is R*t_window
    vrg,=axi5.semilogy(np.asarray(vrg_z),np.asarray(vrg_rate)*twin,linestyle='solid',color='black',linewidth=1.5)
    #pyplot.legend((vrg,),['R-G15'],loc='upper left',fontsize=10,numpoints=1)

    f5.savefig(fm_filen,dpi=300)
    pyplot.close(f5)
    
    #snap_keys=orig_snap_keys
    #fil_keys = orig_fil_keys

    return locals()



def run_random_forest(msF,merF,snap_keys_par,fil_keys_par,rfiter=3,RUN_RF=True,rf_masscut=10.0**(10.5),labelfunc='label_merger1',balancetrain=True):

    full_df=pandas.DataFrame()
    
    for sk,fk in zip(snap_keys_par,fil_keys_par):

        parameters, pcd, pc, pcd = make_pc_dict(msF,sk,fk)
        pc1 = pc.X[:,0].flatten()
        pc2 = pc.X[:,1].flatten()
        pc3 = pc.X[:,2].flatten()
        pc4 = pc.X[:,3].flatten()
        pc5 = pc.X[:,4].flatten()
        pc6 = pc.X[:,5].flatten()
        pc7 = pc.X[:,6].flatten()
        PCs=pandas.DataFrame(pc.X)
        
        asym = get_all_morph_val(msF,sk,fk,'ASYM')
        gini = get_all_morph_val(msF,sk,fk,'GINI')
        m20 = get_all_morph_val(msF,sk,fk,'M20')
        cc = get_all_morph_val(msF,sk,fk,'CC')
        Mstat = get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
        Istat = get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
        Dstat = get_all_morph_val(msF,sk,fk,'MID1_DSTAT')

        sfid = get_all_snap_val(msF,sk,'SubfindID')
        

        
        #latest_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLastGyr')
        #boolean_merger1 = latest_NumMajorMergersLastGyr >= 1.0
        boolean_flag,number = eval(labelfunc+'(merF,sk)')

        
        #this_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLastGyr')
        #boolean_merger2 = this_NumMajorMergersLastGyr >= 1.0
        

        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        log_mstar_mhalo = np.log10( mstar/mhalo )
        sfr = get_all_snap_val(msF,sk,'SFR_Msunperyr')

        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
        
        #set up RF data frame above, run or save input/output for each loop iteration

        rf_dict = {}
        PARAMS_MOD=True
        PARAMS_ONLY=False
        PCS_ONLY=False

        RF_ITER=rfiter

        
        if PCS_ONLY is True:
            gi = np.where(np.isfinite(pc1)*np.isfinite(pc2)*np.isfinite(pc3)*np.isfinite(pc4)*np.isfinite(pc5)*np.isfinite(pc6)*np.isfinite(pc7)*(mstar >= rf_masscut) != 0)[0]
            print(gi.shape, pc1.shape)
            rf_dict['pc1']=pc1[gi]
            rf_dict['pc2']=pc2[gi]
            rf_dict['pc3']=pc3[gi]
            rf_dict['pc4']=pc4[gi]
            rf_dict['pc5']=pc5[gi]
            rf_dict['pc6']=pc6[gi]
            rf_dict['pc7']=pc7[gi]
            rf_dict['mergerFlag']=boolean_flag[gi]
            rf_dict['SubfindID']=sfid[gi]

            cols=['pc1','pc2','pc3','pc4','pc5','pc6','pc7']
            rflabel='pcs'
            label=labelfunc
            
        if PARAMS_ONLY is True:
            gim=np.where(mstar >= rf_masscut)[0]
            gi = np.where(np.isfinite(gini)*np.isfinite(m20)*np.isfinite(asym)*np.isfinite(Mstat)*np.isfinite(Istat)*np.isfinite(Dstat)*np.isfinite(cc)*(mstar >= rf_masscut) != 0)[0]
            print(gi.shape, gim.shape, pc1.shape)
            print(np.sum(boolean_flag[gi]),np.sum(boolean_flag[gim]),np.sum(boolean_flag))
            rf_dict['gini']=gini[gi]
            rf_dict['m20']=m20[gi]
            rf_dict['asym']=asym[gi]
            rf_dict['Mstat']=Mstat[gi]
            rf_dict['Istat']=Istat[gi]
            rf_dict['Dstat']=Dstat[gi]
            rf_dict['cc']=cc[gi]
            rf_dict['mergerFlag']=boolean_flag[gi]
            rf_dict['SubfindID']=sfid[gi]
            rf_dict['Mstar_Msun']=mstar[gi]
            rf_dict['SFR_Msunperyr']=sfr[gi]
            rf_dict['mergerNumber']=number[gi]

            cols=['gini','m20','asym','Mstat','Istat','Dstat','cc']
            rflabel='params'
            label=labelfunc

            print('N mergers per True: ', np.sum(number[gi])/np.sum(boolean_flag[gi]), np.sum(number),np.sum(boolean_flag))
            

            
        if PARAMS_MOD is True:
            gi = np.where(np.isfinite(gini)*np.isfinite(m20)*np.isfinite(asym)*np.isfinite(Mstat)*np.isfinite(Istat)*np.isfinite(Dstat)*np.isfinite(cc)*(mstar >= rf_masscut) != 0)[0]
            print(gi.shape, pc1.shape)
            S_GM20 = SGM20(gini[gi],m20[gi])
            F_GM20 = FGM20(gini[gi],m20[gi])
        
            rf_dict['dGM20']=S_GM20
            rf_dict['fGM20']=F_GM20
            rf_dict['asym']=asym[gi]
            rf_dict['Mstat']=Mstat[gi]
            rf_dict['Istat']=Istat[gi]
            rf_dict['Dstat']=Dstat[gi]
            rf_dict['cc']=cc[gi]
            rf_dict['mergerFlag']=boolean_flag[gi]
            rf_dict['SubfindID']=sfid[gi]
            rf_dict['Mstar_Msun']=mstar[gi]
            rf_dict['SFR_Msunperyr']=sfr[gi]
            rf_dict['mergerNumber']=number[gi]
            
            rf_dict['gini']=gini[gi]
            rf_dict['m20']=m20[gi]
            
            cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc']
            rflabel='paramsmod'
            label=labelfunc



        
        
        if RUN_RF is True:
            if redshift < 4.2:
            
                df=pandas.DataFrame(rf_dict)

                #get a balanced training set
                if balancetrain is True:
                    mergers = df.where(df['mergerFlag']==True).dropna()
                    Nmergers=mergers.shape[0]
                    nonmergers=df.drop(mergers.index)
                    rows=random.sample(list(nonmergers.index),5*int(Nmergers))
                    newdf=mergers.append(nonmergers.ix[rows])
                else:
                    newdf = df
                    mergers = df.where(df['mergerFlag']==True).dropna()
                    Nmergers=mergers.shape[0]

                
                full_df=full_df.append(newdf,ignore_index=True)
                
                print("Running Random Forest... ", sk, fk)
                result, labels, label_probability, rf_objs,roc_results = PyML.randomForestMC(newdf,iterations=RF_ITER,cols=cols,max_leaf_nodes=np.int32(Nmergers*0.5))    #,max_leaf_nodes=30,n_estimators=50
                #result = summary statistics, feature importances (N iterations x N statistics/importances)
                #labels = labels following random forest (N galaxies x N iterations)
                #label_probability = probability of label following random forest (N galaxies x N iterations)

                #saves the output as a file
                if not os.path.lexists('rfoutput'):
                    os.mkdir('rfoutput')
                if not os.path.lexists('rfoutput/'+label):
                    os.mkdir('rfoutput/'+label)

                #is this the right df?
                labels['mergerFlag']=df['mergerFlag']
                label_probability['mergerFlag']=df['mergerFlag']
                labels['SubfindID']=df['SubfindID']
                label_probability['SubfindID']=df['SubfindID']

                
                df.to_pickle('rfoutput/'+label+'/'+rflabel+'_data_{}_{}.pkl'.format(sk,fk))
                newdf.to_pickle('rfoutput/'+label+'/'+rflabel+'_traindata_{}_{}.pkl'.format(sk,fk))

                result.to_pickle('rfoutput/'+label+'/'+rflabel+'_result_{}_{}.pkl'.format(sk,fk))
                labels.to_pickle('rfoutput/'+label+'/'+rflabel+'_labels_{}_{}.pkl'.format(sk,fk))
                label_probability.to_pickle('rfoutput/'+label+'/'+rflabel+'_label_probability_{}_{}.pkl'.format(sk,fk))
                PCs.to_pickle('rfoutput/'+label+'/'+rflabel+'_pc_{}_{}.pkl'.format(sk,fk))
                
                np.save(arr=roc_results,file='rfoutput/'+label+'/'+rflabel+'_roc_{}_{}.npy'.format(sk,fk) )  #.to_pickle('rfoutput/'+label+'/'+rflabel+'_roc_{}_{}.pkl'.format(sk,fk))
                np.save(arr=rf_objs,file='rfoutput/'+label+'/'+rflabel+'_rfobj_{}_{}.npy'.format(sk,fk))



    
    sk='combined'
    fk='subset'
    print("Running Random Forest on combined dataset... ")
    result, labels, label_probability, rf_objs,roc_results = PyML.randomForestMC(full_df,iterations=RF_ITER,cols=cols)    #,max_leaf_nodes=30,n_estimators=50
    #result = summary statistics, feature importances (N iterations x N statistics/importances)
    #labels = labels following random forest (N galaxies x N iterations)
    #label_probability = probability of label following random forest (N galaxies x N iterations)
    
    #saves the output as a file
    if not os.path.lexists('rfoutput'):
        os.mkdir('rfoutput')
    if not os.path.lexists('rfoutput/'+label):
        os.mkdir('rfoutput/'+label)

    labels['mergerFlag']=full_df['mergerFlag']
    label_probability['mergerFlag']=full_df['mergerFlag']
    labels['SubfindID']=full_df['SubfindID']
    label_probability['SubfindID']=full_df['SubfindID']

        
    full_df.to_pickle('rfoutput/'+label+'/'+rflabel+'_data_{}_{}.pkl'.format(sk,fk))
    full_df.to_pickle('rfoutput/'+label+'/'+rflabel+'_traindata_{}_{}.pkl'.format(sk,fk))
        
    result.to_pickle('rfoutput/'+label+'/'+rflabel+'_result_{}_{}.pkl'.format(sk,fk))
    labels.to_pickle('rfoutput/'+label+'/'+rflabel+'_labels_{}_{}.pkl'.format(sk,fk))
    label_probability.to_pickle('rfoutput/'+label+'/'+rflabel+'_label_probability_{}_{}.pkl'.format(sk,fk))
    PCs.to_pickle('rfoutput/'+label+'/'+rflabel+'_pc_{}_{}.pkl'.format(sk,fk))
    
    np.save(arr=roc_results,file='rfoutput/'+label+'/'+rflabel+'_roc_{}_{}.npy'.format(sk,fk) )  #.to_pickle('rfoutput/'+label+'/'+rflabel+'_roc_{}_{}.pkl'.format(sk,fk))
    np.save(arr=rf_objs,file='rfoutput/'+label+'/'+rflabel+'_rfobj_{}_{}.npy'.format(sk,fk))


                
    
    return
    

def make_sfr_radius_mass_plots(msF,merF,rfiter=3):

    for sk,fk in zip(snap_keys,fil_keys):

        parameters, pcd, pc, pcd = make_pc_dict(msF,sk,fk)
        pc1 = pc.X[:,0].flatten()
        pc2 = pc.X[:,1].flatten()
        pc3 = pc.X[:,2].flatten()
        pc4 = pc.X[:,3].flatten()
        pc5 = pc.X[:,4].flatten()
        pc6 = pc.X[:,5].flatten()
        pc7 = pc.X[:,6].flatten()
        PCs=pandas.DataFrame(pc.X)
        
        asym = get_all_morph_val(msF,sk,fk,'ASYM')
        gini = get_all_morph_val(msF,sk,fk,'GINI')
        m20 = get_all_morph_val(msF,sk,fk,'M20')
        cc = get_all_morph_val(msF,sk,fk,'CC')
        Mstat = get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
        Istat = get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
        Dstat = get_all_morph_val(msF,sk,fk,'MID1_DSTAT')

        sfid = get_all_snap_val(msF,sk,'SubfindID')
        
        S_GM20 = SGM20(gini,m20)
        F_GM20 = FGM20(gini,m20)
        
        
        latest_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLastGyr')
        boolean_merger1 = latest_NumMajorMergersLastGyr >= 1.0

        this_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLastGyr')
        boolean_merger2 = this_NumMajorMergersLastGyr >= 1.0
        

        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        sfr = get_all_snap_val(msF,sk,'SFR_Msunperyr')

        log_mstar_mhalo = np.log10( mstar/mhalo )

        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
        
        #set up RF data frame above, run or save input/output for each loop iteration


        
        bins=18

        xlim=[9.7,12.2]
        ylim=[-2.0,3.0]
        rlim=[0.1,1.7]
        
        '''
        plot_filen = 'pc1/sfr_radius_mass_'+sk+'_'+fk+'_pc1.pdf'
        if not os.path.lexists('pc1'):
            os.mkdir('pc1')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval=pc1,vmin=-2,vmax=3,bins=bins)
        gth.make_colorbar(colorobj,title='PC1 morphology',ticks=[-2,-1,0,1,2,3])

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)


        
        plot_filen = 'pc3/sfr_radius_mass_'+sk+'_'+fk+'_pc3.pdf'
        if not os.path.lexists('pc3'):
            os.mkdir('pc3')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval=pc3,vmin=-1,vmax=3,bins=bins)
        gth.make_colorbar(colorobj,title='PC3 morphology',ticks=[-1,0,1,2,3])

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)
    

        
        plot_filen = 'asym/sfr_radius_mass_'+sk+'_'+fk+'_asym.pdf'
        if not os.path.lexists('asym'):
            os.mkdir('asym')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval=asym,vmin=0.0,vmax=0.4,bins=bins)
        gth.make_colorbar(colorobj,title='Asymmetry',ticks=[0.0,0.20,0.40])

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)


        
        plot_filen = 'merger1/sfr_radius_mass_'+sk+'_'+fk+'_merger1.pdf'
        if not os.path.lexists('merger1'):
            os.mkdir('merger1')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval=boolean_merger1,min_bin=3,gridf='fraction_grid',vmin=0.0,vmax=0.5,bins=bins)
        gth.make_colorbar(colorobj,title='fraction major merger',ticks=[0.0,0.25,0.50],format='%.2f')

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)

        
        plot_filen = 'merger3/sfr_radius_mass_'+sk+'_'+fk+'_merger3.pdf'
        if not os.path.lexists('merger3'):
            os.mkdir('merger3')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval=boolean_merger1,min_bin=3,gridf='normed_proportion_grid',vmin=0.0,vmax=1.0,bins=bins)
        gth.make_colorbar(colorobj,title='proportion of major mergers',ticks=[0.0,0.5,1.0],format='%.2f')

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)



        plot_filen = 'mstar_mhalo/sfr_radius_mass_'+sk+'_'+fk+'_mstar_mhalo.pdf'
        if not os.path.lexists('mstar_mhalo'):
            os.mkdir('mstar_mhalo')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval=log_mstar_mhalo,min_bin=3,gridf='median_grid',vmin=-2.0,vmax=-0.5,bins=bins)
        gth.make_colorbar(colorobj,title='median $log_{10} M_*/M_{h}$',ticks=[-2,-1.5,-1,-0.5])

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)




        plot_filen = 'mhalo/sfr_radius_mass_'+sk+'_'+fk+'_mhalo.pdf'
        if not os.path.lexists('mhalo'):
            os.mkdir('mhalo')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval=np.log10(mhalo),min_bin=3,gridf='median_grid',vmin=11.5,vmax=14.0,bins=bins)
        gth.make_colorbar(colorobj,title='median $log_{10} M_{h}$',ticks=[11.5,12.0,13.0,14.0])

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)

        '''

        plot_filen = 'ssfr/sfr_radius_mass_'+sk+'_'+fk+'_ssfr.pdf'
        if not os.path.lexists('ssfr'):
            os.mkdir('ssfr')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval=np.log10(sfr/mstar),min_bin=3,gridf='median_grid',vmin=-11.0,vmax=-7.0,bins=24)
        gth.make_colorbar(colorobj,title='median $log_{10} SFR/M_*$',ticks=[-11,-10,-9,-8,-7])

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)



        plot_filen = 'ssfr_sum/sfr_radius_mass_'+sk+'_'+fk+'_ssfr_sum.pdf'
        if not os.path.lexists('ssfr_sum'):
            os.mkdir('ssfr_sum')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.0)
        colorobj = plot_sfr_radius_mass(msF,merF,sk,fk,f1,xlim=xlim,ylim=ylim,rlim=rlim,Cval={'sfr':sfr,'mstar':mstar},min_bin=3,gridf='summed_logssfr',vmin=-11.0,vmax=-7.0,bins=24)
        gth.make_colorbar(colorobj,title='summed $log_{10} SFR/M_*$',ticks=[-11,-10,-9,-8,-7])

        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)

    
        
    return locals()





def plot_g_m20_cc_a(msF,merF,sk,fk,FIG,xlim,x2lim,ylim,y2lim,Cval,gridf='median_grid',masscut=0.0,**bin_kwargs):

    sfr  = get_all_snap_val(msF,sk,'SFR_Msunperyr')
    mstar  = get_all_snap_val(msF,sk,'Mstar_Msun')
    asym = get_all_morph_val(msF,sk,fk,'ASYM')
    gini = get_all_morph_val(msF,sk,fk,'GINI')
    m20 = get_all_morph_val(msF,sk,fk,'M20')
    cc = get_all_morph_val(msF,sk,fk,'CC')
    
    redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
    
    
    labs=11
    axi = FIG.add_subplot(2,1,1)
    axi.set_ylabel('$C$',labelpad=1,size=labs)
    axi.set_xlabel('$A$',labelpad=1,size=labs)    
    axi.tick_params(axis='both',which='both',labelsize=labs)
    axi.locator_params(nbins=5,prune='both')

    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])


    gi = np.where( (np.isnan(Cval)==False)*(mstar >= masscut))[0]
    
    res = gth.make_twod_grid(axi,asym[gi],cc[gi],{'x':Cval[gi]},gridf,**bin_kwargs)
    


    axi = FIG.add_subplot(2,1,2)
    axi.set_ylabel('$G$',labelpad=1,size=labs)
    axi.set_xlabel('$M_{20}$',labelpad=1,size=labs)
    axi.tick_params(axis='both',which='both',labelsize=labs)
    axi.locator_params(nbins=5,prune='both')
    axi.set_xlim(x2lim[0],x2lim[1])
    axi.set_ylim(y2lim[0],y2lim[1])

    
    kpc_per_arcsec = illcos.kpc_proper_per_arcmin(redshift).value/60.0
    size_pixels = get_all_morph_val(msF,sk,fk,'RP')
    pix_arcsec  = get_all_morph_val(msF,sk,fk,'PIX_ARCSEC')
    size_kpc = size_pixels*pix_arcsec*kpc_per_arcsec


    
    axi,colorobj = gth.make_twod_grid(axi,m20[gi],gini[gi],{'x':Cval[gi]},gridf,flipx=True,**bin_kwargs)
    axi.annotate('z = {:3.1f}'.format(redshift) ,xy=(0.80,0.05),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    axi.annotate(fk                             ,xy=(0.80,0.15),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    if masscut > 0.0:
        axi.annotate('$log_{10} M_{*} > $'+'{:4.1f}'.format(np.log10(masscut)),xy=(0.80,0.25),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    
    return colorobj




def make_morphology_plots(msF,merF,vardict=None):

    for sk,fk in zip(snap_keys,fil_keys):
        parameters, pcd, pc, pcd = make_pc_dict(msF,sk,fk)
        pc1 = pc.X[:,0].flatten()
        pc2 = pc.X[:,1].flatten()
        pc3 = pc.X[:,2].flatten()
        pc4 = pc.X[:,3].flatten()
        pc5 = pc.X[:,4].flatten()
        pc6 = pc.X[:,5].flatten()
        pc7 = pc.X[:,6].flatten()
        PCs=pandas.DataFrame(pc.X)
        
        asym = get_all_morph_val(msF,sk,fk,'ASYM')
        gini = get_all_morph_val(msF,sk,fk,'GINI')
        m20 = get_all_morph_val(msF,sk,fk,'M20')
        cc = get_all_morph_val(msF,sk,fk,'CC')
        Mstat = get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
        Istat = get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
        Dstat = get_all_morph_val(msF,sk,fk,'MID1_DSTAT')

        S_GM20 = SGM20(gini,m20)
        F_GM20 = FGM20(gini,m20)
        
        
        latest_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLastGyr')
        boolean_merger1 = latest_NumMajorMergersLastGyr >= 1.0
        latest_NumMinorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLastGyr')
        boolean_merger4 = latest_NumMinorMergersLastGyr >= 1.0

        this_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLastGyr')
        boolean_merger2 = this_NumMajorMergersLastGyr >= 1.0

        boolean_merger_window500_both = label_merger_window500_both(merF,sk)

        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        log_mstar_mhalo = np.log10( mstar/mhalo )

        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]

        
        bins=14

        xlim=[0.0,1.0]
        x2lim=[-0.5,-2.7]
        ylim=[0.4,5.0]
        y2lim=[0.35,0.70]

        masscut = 10.0**(10.5)
        numbers = [3,30]
        
        plot_filen = 'pc1/morphology_'+sk+'_'+fk+'_pc1.pdf'
        if not os.path.lexists('pc1'):
            os.mkdir('pc1')
        print(plot_filen)
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=pc1,vmin=-2,vmax=3,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='PC1 morphology',ticks=[-2,-1,0,1,2,3])
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)

        plot_filen = 'pc3/morphology_'+sk+'_'+fk+'_pc1.pdf'
        if not os.path.lexists('pc3'):
            os.mkdir('pc3')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=pc3,vmin=-1,vmax=3,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='PC3 morphology',ticks=[-1,0,1,2,3])
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)


        plot_filen = 'asym/morphology_'+sk+'_'+fk+'_pc1.pdf'
        if not os.path.lexists('asym'):
            os.mkdir('asym')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=asym,vmin=0.0,vmax=0.4,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='Asymmetry',ticks=[0.0,0.2,0.4])
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)


        plot_filen = 'merger1/morphology_'+sk+'_'+fk+'_merger1.pdf'
        if not os.path.lexists('merger1'):
            os.mkdir('merger1')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=boolean_merger1,min_bin=3,gridf='fraction_grid',vmin=0.0,vmax=0.5,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='fraction major merger',ticks=[0.0,0.25,0.50],format='%.2f')
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)

        
        plot_filen = 'merger2/morphology_'+sk+'_'+fk+'_merger2.pdf'
        if not os.path.lexists('merger2'):
            os.mkdir('merger2')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=boolean_merger2,min_bin=3,gridf='fraction_grid',vmin=0.0,vmax=0.5,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='fraction past major merger',ticks=[0.0,0.25,0.50],format='%.2f')
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)


        plot_filen = 'merger3/morphology_'+sk+'_'+fk+'_merger3.pdf'
        if not os.path.lexists('merger3'):
            os.mkdir('merger3')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=boolean_merger1,min_bin=3,gridf='normed_proportion_grid',vmin=0.0,vmax=1.0,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='proportion of major mergers',ticks=[0.0,0.5,1.0],format='%.2f')
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)


        plot_filen = 'merger4/morphology_'+sk+'_'+fk+'_merger4.pdf'
        if not os.path.lexists('merger4'):
            os.mkdir('merger4')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=boolean_merger4,min_bin=3,gridf='fraction_grid',vmin=0.0,vmax=0.5,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='fraction minor+major merger',ticks=[0.0,0.25,0.50],format='%.2f')
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)
        


        plot_filen = 'merger_window500_both/morphology_'+sk+'_'+fk+'_merger_window500_both.pdf'
        if not os.path.lexists('merger_window500_both'):
            os.mkdir('merger_window500_both')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=boolean_merger_window500_both,min_bin=3,gridf='fraction_grid',vmin=0.0,vmax=0.5,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='fraction minor+major merger',ticks=[0.0,0.25,0.50],format='%.2f')
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)
        
        
        
        plot_filen = 'mstar_mhalo/morphology_'+sk+'_'+fk+'_mstar_mhalo.pdf'
        if not os.path.lexists('mstar_mhalo'):
            os.mkdir('mstar_mhalo')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=log_mstar_mhalo,min_bin=3,gridf='median_grid',vmin=-2.0,vmax=-0.5,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='median $log_{10} M_*/M_{h}$',ticks=[-2,-1.5,-1,-0.5])
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)


        plot_filen = 'mhalo/morphology_'+sk+'_'+fk+'_mhalo.pdf'
        if not os.path.lexists('mhalo'):
            os.mkdir('mhalo')
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.22)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=np.log10(mhalo),min_bin=3,gridf='median_grid',vmin=11.5,vmax=14.0,bins=bins,masscut=masscut,numbers=numbers)
        gth.make_colorbar(colorobj,title='median $log_{10} M_{h}$',ticks=[11.5,12.0,13.0,14.0])
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)

        

        
    return locals()






def make_merger_images(msF,merF,bd='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/',rflabel='paramsmod',rf_masscut=0.0,labelfunc='label_merger1'):


    for j,sk in enumerate(im_snap_keys):

        plotdir = 'images/'+labelfunc
        plot_filen = plotdir+'/mergers_'+sk+'.pdf'
        
        if not os.path.lexists(plotdir):
            os.mkdir(plotdir)
        
        f1 = pyplot.figure(figsize=(12.0,10.0), dpi=600)
        pyplot.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,wspace=0.0,hspace=0.0)

        N_columns = 12
        N_rows = 10
        N_pix = im_npix[j]  #140
    
    

        #pick filter to select PC1 values
        r_fk = im_fil_keys[sk]['r'][0]
        g_fk = im_fil_keys[sk]['g'][0]
        b_fk = im_fil_keys[sk]['b'][0]

        #parameters, pcd, pc, pcd = make_pc_dict(msF,sk,r_fk)
        #pc1 = pc.X[:,0].flatten()  #all fk's pc1 values for this snapshot
        mag = get_all_morph_val(msF,sk,r_fk,'MAG')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        
        r_imf = np.asarray(get_all_morph_val(msF,sk,r_fk,'IMFILES'),dtype='str')
        g_imf = np.asarray(get_all_morph_val(msF,sk,g_fk,'IMFILES'),dtype='str')
        b_imf = np.asarray(get_all_morph_val(msF,sk,b_fk,'IMFILES'),dtype='str')

        #select indices  whose images we want

        fk = im_rf_fil_keys[j]

        forest_imf = np.asarray(get_all_morph_val(msF,sk,fk,'IMFILES'),dtype='str')
        
        asym = get_all_morph_val(msF,sk,fk,'ASYM')
        gini = get_all_morph_val(msF,sk,fk,'GINI')
        m20 = get_all_morph_val(msF,sk,fk,'M20')
        cc = get_all_morph_val(msF,sk,fk,'CC')
        Mstat = get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
        Istat = get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
        Dstat = get_all_morph_val(msF,sk,fk,'MID1_DSTAT')

        sfid = get_all_snap_val(msF,sk,'SubfindID')
        
        S_GM20 = SGM20(gini,m20)
        F_GM20 = FGM20(gini,m20)


        print(Istat.shape,mstar.shape,sfid.shape)# looks like I synced the array shapes -- good!
        gi = np.where(np.isfinite(S_GM20)*np.isfinite(F_GM20)*np.isfinite(asym)*np.isfinite(Mstat)*np.isfinite(Istat)*np.isfinite(Dstat)*np.isfinite(cc)*(mstar >= rf_masscut) != 0)[0]


        all_sfids = get_all_snap_val(msF,sk,'SubfindID')

        latest_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLastGyr')
        boolean_merger1 = latest_NumMajorMergersLastGyr >= 1.0
        latest_NumMinorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLastGyr')
        boolean_merger4 = latest_NumMinorMergersLastGyr >= 1.0

        this_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLastGyr')
        boolean_merger2 = this_NumMajorMergersLastGyr >= 1.0

        mlpid_match = get_mergerinfo_val(merF,sk,'MainLeafID_match')
        
        print(boolean_merger4.shape,mlpid_match.shape)   #I combined these also!
        
        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        log_mstar_mhalo = np.log10( mstar/mhalo )


        
        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
        ins = fk.split('-')[0]
        if redshift > 4.2:
            continue

        rfdata = 'rfoutput/'+labelfunc+'/'
        
        data_file = rfdata+rflabel+'_data_cut_{}_{}.pkl'.format(sk,fk)
        result_file= rfdata+rflabel+'_result_cut_{}_{}.pkl'.format(sk,fk)
        labels_file = rfdata+rflabel+'_labels_cut_{}_{}.pkl'.format(sk,fk)
        prob_file = rfdata+rflabel+'_label_probability_cut_{}_{}.pkl'.format(sk,fk)
        pcs_file= rfdata+rflabel+'_pc_cut_{}_{}.pkl'.format(sk,fk)

        rf_data = np.load(data_file,encoding='bytes')
        rf_asym = rf_data['asym'].values
        rf_flag = rf_data['mergerFlag'].values
        #rf_sgm20 = rf_data['dGM20'].values
        rf_cc = rf_data['cc'].values
        
        asym_classifier = rf_asym > 0.25
        asym_com,asym_ppv = simple_classifier_stats(asym_classifier,rf_flag)

        #sgm_classifier = rf_sgm20 > 0.10
        #sgm_com,sgm_ppv = simple_classifier_stats(sgm_classifier,rf_flag)

        #three_classifier = np.logical_or(rf_asym > 0.35,rf_sgm20 > 0.10)
        #three_com,three_ppv = simple_classifier_stats(three_classifier,rf_flag)
        

        result = np.load(result_file)
        completeness = np.median(result['completeness'].values)
        ppv = np.median(result['ppv'].values)
        
        #probs = data frame with N random forest iterations, mergerFlag, and SubfindID
        probs = np.load(prob_file)
        
        iters = result['completeness'].values.shape[0]

        probs_subframe = probs[probs.keys()[0:iters]]

        average_prob = probs_subframe.apply(np.mean,axis=1)
        flags = probs['mergerFlag']
        rf_sfids = probs['SubfindID']
        
        print(flags.shape,gi.shape)


        good_sfid = all_sfids[gi]
        assert(np.all(good_sfid==rf_sfids)==True)
        
        print('redshift: {:3.1f}   filter: {:15s}  # iterations:  {:8d}   RF sample size: {:8d}   # True mergers: {} '.format(redshift,fk,iters,rf_asym.shape[0],np.sum(rf_flag)))
        
        #continue
        
        #pc1_bins = np.asarray([-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5])
        mstar_bins = 10.0**np.asarray([10.0,10.1,10.2,10.3,10.4,10.5,10.7,10.9,11.1,11.3])

        
        nprob,edges = np.histogram(average_prob,bins=10)
        print(nprob, edges)

        
        for i,pb in enumerate(edges[1:]):

            pi = np.where(np.logical_and(average_prob <= edges[i+1], average_prob > edges[i]))[0]  #index into probs

            #select up to 8 of these to show
            sortpi = np.argsort(average_prob[pi])  #index into pi
            nplot = np.min([pi.shape[0],N_columns])
            plotpi = sortpi[0:nplot] #sortpi[-nplot:]

            for k,pii in enumerate(plotpi):
                
                im_i = gi[pi[pii]]  #gi indexes image array to RF inputs ; pi indexes this bin ; pii is this exact one
                morph_sfid = all_sfids[gi[pi[pii]]]
                rf_sfid = rf_sfids[pi[pii]]
                assert(morph_sfid==rf_sfid)
                
                this_mer4 = boolean_merger4[gi[pi[pii]]]
                this_match = mlpid_match[gi[pi[pii]]]
                
                r_im = bd+sk+'/'+r_imf[im_i]
                g_im = bd+sk+'/'+g_imf[im_i]
                b_im = bd+sk+'/'+b_imf[im_i]
                rf_im = bd+sk+'/'+forest_imf[im_i]
                
                r = pyfits.open(r_im)[0].data
                g = pyfits.open(g_im)[0].data
                b = pyfits.open(b_im)[0].data
                mid = np.int64(r.shape[0]/2)
                delt=np.int64(N_pix/2)
                
                r = r[mid-delt:mid+delt,mid-delt:mid+delt]
                g = g[mid-delt:mid+delt,mid-delt:mid+delt]
                b = b[mid-delt:mid+delt,mid-delt:mid+delt]

                r_header = pyfits.open(rf_im)[0].header
                this_camnum_int = r_header['CAMERA']
                
                this_prob = average_prob[pi[pii]]
                
                axi = f1.add_subplot(N_rows,N_columns,N_columns*(i+1)-k)
                axi.set_xticks([]) ; axi.set_yticks([])
            
                alph=0.2 ; Q=8.0
            
                rgbthing = make_color_image.make_interactive(b,g,r,alph,Q)
                axi.imshow(rgbthing,interpolation='nearest',aspect='auto',origin='lower')


                #need to re-center and zoom subsequent plots
                axi = overplot_morph_data(axi,rf_im,mid,delt)

                
                this_flag = flags[pi[pii]]
                
                if this_flag==True:
                    fcolor = 'Green'
                else:
                    fcolor = 'Red'


                if this_flag==True and this_match==True:
                    axi.annotate('$unique$',(0.75,0.10),xycoords='axes fraction',ha='center',va='center',color=fcolor,size=8)
                elif this_flag==True and this_match==False:
                    axi.annotate('$infall$',(0.75,0.10),xycoords='axes fraction',ha='center',va='center',color=fcolor,size=8)
                elif this_flag==False:
                    axi.annotate('${:5s}$'.format(str(this_flag)),(0.75,0.10),xycoords='axes fraction',ha='center',va='center',color=fcolor,size=8)
                    
                    
                axi.annotate('${:4.2f}$'.format(this_prob),(0.25,0.10),xycoords='axes fraction',ha='center',va='center',color='White',size=7)
                        
                axi.annotate('${:7d}$'.format(rf_sfid),(0.25,0.90),xycoords='axes fraction',ha='center',va='center',color='White',size=6)

                axi.annotate('${:2d}$'.format(this_camnum_int),(0.25,0.82),xycoords='axes fraction',ha='center',va='center',color='White',size=3 )
                
                if this_flag==False and this_mer4==True:
                    axi.annotate('$M$',(0.85,0.15),xycoords='axes fraction',ha='center',va='center',color='Green',size=6)

                    
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)
    
    return 0


def overplot_morph_data(axi,rf_im,mid,delt,lw=0.1):

    
    hlist = pyfits.open(rf_im)

    try:
        saveseg_hdu = hlist['SEGMAP']
    except KeyError as e:
        saveseg_hdu = None
        
    try:
        tbhdu = hlist['PhotUtilsMeasurements']
    except KeyError as e:
        tbhdu = None

    try:
        mhdu = hlist['LotzMorphMeasurements']
    except KeyError as e:
        mhdu = None

    try:
        ap_seghdu = hlist['APSEGMAP']
    except KeyError as e:
        ap_seghdu = None


    
    #plot initial photutils segmentation map contour
    if saveseg_hdu is not None:
        segmap = saveseg_hdu.data[mid-delt:mid+delt,mid-delt:mid+delt]
        clabel = saveseg_hdu.header['CLABEL']
        segmap_masked = np.where(segmap==clabel,segmap,np.zeros_like(segmap))
        axi.contour(np.transpose(segmap_masked), (clabel-0.0001,), linewidths=lw, colors=('DodgerBlue',))


    if tbhdu is not None:
        centrsize = 1
        #plot centroid
        #axi.plot([tbhdu.header['POS0']],[tbhdu.header['POS1']],'o',color='DodgerBlue',markersize=centrsize,alpha=0.6,mew=0)
        #axi.plot([tbhdu.header['XCENTR']],[tbhdu.header['YCENTR']],'o',color='Yellow',markersize=centrsize,alpha=0.6,mew=0)

        
    #plot asymmetry center and elliptical (and circular) petrosian radii ???
    if mhdu is not None:
        if mhdu.header['FLAG']==0:
            axc = mhdu.header['AXC']
            ayc = mhdu.header['AYC']
            rpe = mhdu.header['RPE']
            elongation = mhdu.header['ELONG']
            position = (axc, ayc)
            a = rpe
            b = rpe/elongation
            theta = mhdu.header['ORIENT']
            aperture = photutils.EllipticalAperture(position, a, b, theta=theta)
            #aperture.plot(color='Orange', alpha=0.4, ax=axi,linewidth=1)
            #axi.plot([ayc],[axc],'+',color='Orange',markersize=centrsize,mew=0.1)


    #plot petrosian morphology segmap in different linestyle
    if ap_seghdu is not None:
        ap_segmap = ap_seghdu.data[mid-delt:mid+delt,mid-delt:mid+delt]
        axi.contour(np.transpose(ap_segmap), (10.0-0.0001,), linewidths=lw,colors='Orange')

    hlist.close()
    

    return axi


def make_pc1_images(msF,merF,bd='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/'):


    plot_filen = 'images/pc1_deciles.pdf'
    if not os.path.lexists('images'):
        os.mkdir('images')
    f1 = pyplot.figure(figsize=(8.0,10.0), dpi=600)
    pyplot.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,wspace=0.0,hspace=0.0)

    N_columns = 8
    N_rows = 10
    N_pix = 80
    
    
    for j,sk in enumerate(im_snap_keys):

        #pick filter to select PC1 values
        r_fk = im_fil_keys[sk]['r'][0]
        g_fk = im_fil_keys[sk]['g'][0]
        b_fk = im_fil_keys[sk]['b'][0]

        parameters, pcd, pc, pcd = make_pc_dict(msF,sk,r_fk)
        pc1 = pc.X[:,0].flatten()  #all fk's pc1 values for this snapshot
        mag = get_all_morph_val(msF,sk,r_fk,'MAG')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        
        r_imf = get_all_morph_val(msF,sk,r_fk,'IMFILES')
        g_imf = get_all_morph_val(msF,sk,g_fk,'IMFILES')
        b_imf = get_all_morph_val(msF,sk,b_fk,'IMFILES')

        #select indices  whose images we want
        pc1_bins = np.asarray([-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5])
        mstar_bins = 10.0**np.asarray([10.0,10.1,10.2,10.3,10.4,10.5,10.7,10.9,11.1,11.3])
        
        for i,pb in enumerate(pc1_bins):
            pi = np.where(np.logical_and( pc1 < pb+0.5, pc1 >= pb) )[0]
            
            msi = pi #np.where(np.logical_and( mstar < pb+1.0e9, mstar >=pb))[0]
            
            if msi.shape[0]==0:
                continue
            
            mi = -1#np.argmin(mag[msi])
            
            r_im = bd+sk+'/'+r_imf[msi[mi]]
            g_im = bd+sk+'/'+g_imf[msi[mi]]
            b_im = bd+sk+'/'+b_imf[msi[mi]]
            
            r = pyfits.open(r_im)[0].data
            g = pyfits.open(g_im)[0].data
            b = pyfits.open(b_im)[0].data
            print(r_im,r.shape)
            mid = r.shape[0]/2

            r = r[mid-N_pix/2:mid+N_pix/2,mid-N_pix/2:mid+N_pix/2]
            g = g[mid-N_pix/2:mid+N_pix/2,mid-N_pix/2:mid+N_pix/2]
            b = b[mid-N_pix/2:mid+N_pix/2,mid-N_pix/2:mid+N_pix/2]

            
            axi = f1.add_subplot(N_rows,N_columns,N_columns*(i+1)-j)
            axi.set_xticks([]) ; axi.set_yticks([])
            
            alph=0.2 ; Q=8.0
            
            rgbthing = make_color_image.make_interactive(b,g,r,alph,Q)
            axi.imshow(rgbthing,interpolation='nearest',aspect='auto',origin='lower')
            
            
    f1.savefig(plot_filen,dpi=300)
    pyplot.close(f1)
    
    return 0



def make_morphology_evolution_plots(msF,merF):

    final_key='snapshot_068'

    #this is the list of subfindIDs of the MMPB of the given subhalo
    tree_sfid_grid=merF['mergerinfo'][final_key]['Tree_SFID_grid'].value
    final_sfids=merF['mergerinfo'][final_key]['SubfindID'].value
    snap_keys=list(merF['mergerinfo'].keys())
    snap_keys.append('snapshot_135')
    print(snap_keys)

    tree_df=pandas.DataFrame(data=tree_sfid_grid,columns=snap_keys)

    evol_keys=np.flipud(np.asarray(['snapshot_103','snapshot_085','snapshot_075','snapshot_068','snapshot_064','snapshot_060','snapshot_054']))
    filters = np.flipud(np.asarray(['ACS-F606W','ACS-F814W','NC-F115W','NC-F150W','NC-F150W','NC-F200W','NC-F277W']))

    fil_keys = pandas.DataFrame(data=filters.reshape(1,7),columns=evol_keys)

    print(fil_keys)


    sub_tree_df=tree_df[evol_keys]
    m20_df = pandas.DataFrame()
    mstar_df = pandas.DataFrame()
    redshifts=[]
    ages=[]

    for ek in evol_keys:
        snap_int=int(ek[-3:])
        redshift =gsu.redshift_from_snapshot(snap_int)
        redshifts.append(redshift)
        ages.append(gsu.age_at_snap(snap_int))

        print(ek,snap_int,redshift)
        fk = fil_keys[ek].values[0]
        print(fk)

        sfid_search =  np.asarray(sub_tree_df[ek].values)

        m20_values= np.asarray(get_all_morph_val(msF,ek,fk,'M20'))
        sfid_values=np.asarray(get_all_snap_val(msF,ek,'SubfindID'))
        mstar_values = get_all_snap_val(msF,ek,'Mstar_Msun')

        print(m20_values.shape,sfid_values.shape)

        m20_array=np.empty(shape=0)
        mstar_array = np.empty(shape=0)

        for i,search_sfid in enumerate(sfid_search):

            bi= sfid_values==search_sfid

            if np.sum(bi)==4:
                m20_array=np.append(m20_array,m20_values[bi])
                mstar_array=np.append(mstar_array,mstar_values[bi])
            else:
                m20_array = np.append(m20_array,np.asarray([0.0,0.0,0.0,0.0]))
                mstar_array = np.append(mstar_array,np.asarray([0.0,0.0,0.0,0.0]))

        m20_df[ek]=m20_array
        mstar_df[ek]=mstar_array
        print(np.sum(mstar_df[ek]),np.sum(mstar_df[ek] > 0.0))

    print(m20_df.shape,mstar_df.shape)

    m20_new = m20_df.dropna()

    mstar_new = mstar_df.loc[m20_new.index.values][:]
    print(m20_new.shape,mstar_new.shape)

    mstar_sort_index = m20_new['snapshot_103'].sort_values(ascending=True).index.values

    #sorted by final stellar mass
    mstar_sorted = mstar_new.loc[mstar_sort_index,:]
    m20_sorted = m20_new.loc[mstar_sort_index,:]


    plot_filen = 'evolution/m20_evolution.pdf'
    if not os.path.lexists('evolution'):
        os.mkdir('evolution')
        
    f1 = pyplot.figure(figsize=(10.0,10.0), dpi=600)
    pyplot.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.98,wspace=0.0,hspace=0.0)
    axi=f1.add_subplot(1,1,1)

    N=np.sum(mstar_new > 0.0)
    totalM = np.sum(mstar_new)


    axi.plot(ages,N,'ok')

    tstart=[1.0]
    for a in ages:
        tstart.append(a)
    
    prek=np.append('snapshot_054',evol_keys)
    print(prek)

    for ek,age,z,ts,pk in zip(evol_keys,ages,redshifts,tstart,prek):
        print(ek,age,z,ts,pk)
        Xarr=np.asarray([ts,age])

        #summed=np.cumsum(mstar_sorted[final_key])
        #Yarr=np.asarray(summed)
        sum2=np.cumsum(np.ones_like( mstar_sorted[final_key]) )
        sum1=np.cumsum(np.ones_like( mstar_sorted[final_key]) )
        Yarr=np.transpose(np.asarray([sum1,sum2]))

        Carr = np.asarray([m20_sorted[ek],m20_sorted[ek]])

        print(Carr.shape,Xarr.shape,Yarr.shape)

        Zm = ma.masked_where(np.logical_or(Carr >= 0.0,np.isnan(Carr) ) , Carr)

        obj=axi.pcolormesh(Xarr,Yarr,np.transpose(Zm),vmin=-2.5,vmax=-1.5,edgecolors='None',shading='flat')
    
    gth.make_colorbar(obj,title='$M_{20}$',ticks=[-2.5,-2.0,-1.5],loc=[0.15,0.90,0.40,0.07],fontsize=20)


    sp=25
    axi.set_ylim(0,3.5e3)
    axi.set_xlim(1,9)
    axi.set_ylabel('$N$',size=sp)
    axi.set_xlabel('cosmic time (Gyr)',size=sp)
    axi.tick_params(labelsize=sp)

    f1.savefig(plot_filen,dpi=600)
    pyplot.close(f1)


    return mstar_sorted






def plot_sfr_mass(msF,merF,sk,fk,FIG,Cval,gridf='median_grid',sfr=None,mstar=None,nx=4,ny=2,ii=1,i=1,redshift=0.0,skipthis=False,**bin_kwargs):

    if sfr is None:
        sfr  = get_all_snap_val(msF,sk,'SFR_Msunperyr')
    if mstar is None:
        mstar  = get_all_snap_val(msF,sk,'Mstar_Msun')
    
    bins=18
    labs=10

    xlim=[9.7,11.9]
    ylim=[-1.5,3.2]

    axi=FIG.add_subplot(ny,nx,ii)
    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])
        
        

    #if given array, use it
    if type(Cval)==np.ndarray:
        gi = np.where(np.isfinite(Cval)==True)[0]
        data_dict = {'x':Cval[gi]}
    else:
        #otherwise, loop over dict keys and send to plotter
        #this assumes that all values are finite!
        gi = np.arange(mstar.shape[0])
        data_dict = copy.copy(Cval)

    if skipthis is True:
        pass
        res=None
        colorobj=None
    else:
        res,colorobj = gth.make_twod_grid(axi,np.log10(mstar[gi]),np.log10(sfr[gi]),data_dict,gridf,bins=bins,**bin_kwargs)


    anny=0.92

    axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    axi.annotate(fk                             ,xy=(0.30,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)


    if i==2*nx:
        axi.set_ylabel('$log_{10}\ SFR [M_{\odot}/yr]$',labelpad=1,size=labs)
    if i==nx:
        axi.set_xlabel('$log_{10}\ M_{*} [M_{\odot}]$',labelpad=1,size=labs)
    if i > nx:
        axi.set_xticklabels([])
    if i != nx and i !=2*nx:
        axi.set_yticklabels([])


    return colorobj





def plot_g_m20(msF,merF,sk,fk,FIG,Cval,gridf='median_grid',g=None,m20=None,nx=4,ny=2,ii=1,i=1,redshift=0.0,skipthis=False,**bin_kwargs):

    if g is None:
        g  = get_all_morph_val(msF,sk,fk,'GINI')
    if m20 is None:
        m20  = get_all_morph_val(msF,sk,fk,'M20')
    
    bins=18
    labs=10

    xlim=[-0.4,-2.7]
    ylim=[0.35,0.69]

    axi=FIG.add_subplot(ny,nx,ii)
    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])
        
        

    #if given array, use it
    if type(Cval)==np.ndarray:
        gi = np.where(np.isfinite(Cval)==True)[0]
        data_dict = {'x':Cval[gi]}
    else:
        #otherwise, loop over dict keys and send to plotter
        #this assumes that all values are finite!
        gi = np.arange(m20.shape[0])
        data_dict = copy.copy(Cval)

    if skipthis is True:
        pass
        res=None
        colorobj=None
    else:
        res,colorobj = gth.make_twod_grid(axi,m20[gi],g[gi],data_dict,gridf,bins=bins,flipx=True,**bin_kwargs)


    anny=0.92

    axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    axi.annotate(fk                             ,xy=(0.30,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)


    if i==2*nx:
        axi.set_ylabel('$G$',labelpad=1,size=labs)
    if i==nx:
        axi.set_xlabel('$M_{20}$',labelpad=1,size=labs)
    if i > nx:
        axi.set_xticklabels([])
    if i != nx and i !=2*nx:
        axi.set_yticklabels([])


    return colorobj






def make_structure_plots(msF,merF,name='pc1',varname=None,bartitle='median PC1',vmin=-2,vmax=3,barticks=[-2,-1,0,1,2,3],
                         rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',rflabel='params',gridf='median_grid',log=False):
    
    if varname is None:
        varname=name

    #start with PC1 or M20
    plot_filen = 'structure/'+labelfunc+'/sfr_mstar_'+name+'_evolution.pdf'
    if not os.path.lexists('structure'):
        os.mkdir('structure')
    if not os.path.lexists('structure/'+labelfunc):
        os.mkdir('structure/'+labelfunc)

    f1 = pyplot.figure(figsize=(6.5,4.0), dpi=300)
    pyplot.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)


    #G-M20
    plot2_filen = 'structure/'+labelfunc+'/G_M20_'+name+'_evolution.pdf'

    f2 = pyplot.figure(figsize=(6.5,4.0), dpi=300)
    pyplot.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)


    nx=4
    ny=2
    
    i=0
    for sk,fk in zip(snap_keys,fil_keys):
        i=i+1

        parameters, pcd, pc, pcd = make_pc_dict(msF,sk,fk)
        pc1 = pc.X[:,0].flatten()
        pc2 = pc.X[:,1].flatten()
        pc3 = pc.X[:,2].flatten()
        pc4 = pc.X[:,3].flatten()
        pc5 = pc.X[:,4].flatten()
        pc6 = pc.X[:,5].flatten()
        pc7 = pc.X[:,6].flatten()
        PCs=pandas.DataFrame(pc.X)
        
        asym = get_all_morph_val(msF,sk,fk,'ASYM')
        gini = get_all_morph_val(msF,sk,fk,'GINI')
        m20 = get_all_morph_val(msF,sk,fk,'M20')
        cc = get_all_morph_val(msF,sk,fk,'CC')
        Mstat = get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
        Istat = get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
        Dstat = get_all_morph_val(msF,sk,fk,'MID1_DSTAT')

        sfid = get_all_snap_val(msF,sk,'SubfindID')
        
        S_GM20 = SGM20(gini,m20)
        F_GM20 = FGM20(gini,m20)
        

        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        sfr = get_all_snap_val(msF,sk,'SFR_Msunperyr')

        log_mstar_mhalo = np.log10( mstar/mhalo )

        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
        

        rfdata = 'rfoutput/'+labelfunc+'/'
        
        data_file = rfdata+rflabel+'_traindata_{}_{}.pkl'.format(sk,fk)
        result_file= rfdata+rflabel+'_result_{}_{}.pkl'.format(sk,fk)
        labels_file = rfdata+rflabel+'_labels_{}_{}.pkl'.format(sk,fk)
        prob_file = rfdata+rflabel+'_label_probability_{}_{}.pkl'.format(sk,fk)
        pcs_file= rfdata+rflabel+'_pc_{}_{}.pkl'.format(sk,fk)
        roc_file= rfdata+rflabel+'_roc_{}_{}.npy'.format(sk,fk)

        if (varname=='rfprob' or varname=='rf_flag' or varname=='rf_class'):
            if redshift > 4.2:
                skipthis=True
                sfr=None
                mstar=None
                g=gini
                m20=m20
            else:
                skipthis=False
                rf_data = np.load(data_file,encoding='bytes')
                rf_asym = rf_data['asym'].values
                rf_flag = rf_data['mergerFlag'].values
                #rf_sgm20 = rf_data['dGM20'].values
                rf_cc = rf_data['cc'].values
                rf_mstar = rf_data['Mstar_Msun'].values
                rf_sfr = rf_data['SFR_Msunperyr'].values
                rf_gini=rf_data['gini'].values
                rf_m20=rf_data['m20'].values

                result = np.load(result_file)
                completeness = np.median(result['completeness'].values)
                ppv = np.median(result['ppv'].values)
                probs = np.load(prob_file)
                iters = result['completeness'].values.shape[0]

                rfprob=np.asarray( np.mean(probs[[0,1,2,3,4]],axis=1))
                
                rf_class = rfprob > 0.4

                roc = (np.load(roc_file)).all()
                #axi3.plot(roc['fpr'],roc['tpr'])

                sfr=rf_sfr
                mstar=rf_mstar
                g=rf_gini
                m20=rf_m20
        else:
            skipthis=False
            sfr=None
            mstar=None
            g=gini
            m20=m20

        if log is True:
            Cval=np.log10( locals()[varname] )
        else:            
            Cval = locals()[varname]
            
        ii=9-i*1

        colorobj=plot_sfr_mass(msF,merF,sk,fk,f1,Cval,vmin=vmin,vmax=vmax,sfr=sfr,mstar=mstar,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis)

        colorobj=plot_g_m20(msF,merF,sk,fk,f2,Cval,vmin=vmin,vmax=vmax,g=g,m20=m20,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis)

        if i==1:
            the_colorobj=copy.copy(colorobj)


    
    gth.make_colorbar(the_colorobj,f1,title=bartitle,ticks=barticks,loc=[0.12,0.65,0.17,0.03],format='%1d')

    gth.make_colorbar(the_colorobj,f2,title=bartitle,ticks=barticks,loc=[0.12,0.65,0.17,0.03],format='%1d')



    f1.savefig(plot_filen,dpi=300)
    pyplot.close(f1)

    f2.savefig(plot2_filen,dpi=300)
    pyplot.close(f2)


        
    #also show PC3 or Asym

    #if RF results exist, also make that one

    
def make_all_structures(msF,merF,rf_labelfunc,rflabel):

    res = make_structure_plots(msF,merF,name='pc1',bartitle='median PC1',vmin=-2,vmax=3,barticks=[-2,-1,0,1,2,3],labelfunc=rf_labelfunc,rflabel=rflabel)
    res = make_structure_plots(msF,merF,name='pc3',bartitle='median PC3',vmin=-1,vmax=2,barticks=[-1,0,1,2],labelfunc=rf_labelfunc,rflabel=rflabel)
                
    res = make_structure_plots(msF,merF,name='rfprob',bartitle='$log_{10} <P_{RF}>$',vmin=-2,vmax=0,barticks=[-2,-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=True,rflabel=rflabel)
                
    res = make_structure_plots(msF,merF,name='rf_flag',bartitle='$log_{10} f_{merger}$',gridf='log_fraction_grid',vmin=-2,vmax=0,barticks=[-2,-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel)
                
    res = make_structure_plots(msF,merF,name='rf_class',bartitle='$log_{10} f_{RF}(0.4)$',gridf='log_fraction_grid',vmin=-2,vmax=0,barticks=[-2,-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel)
                
    res = make_structure_plots(msF,merF,name='rf_prop',varname='rf_flag',bartitle='merger proportion',gridf='normed_proportion_grid',vmin=0,vmax=1,barticks=[0,1],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel)
                

    return


def make_filter_images(msF,merF,snapkey='snapshot_068',sfid=105747,alph=2.0,Q=2.0,cams='00'):
    filen='images/'+snapkey+'/filter_images_'+snapkey+'_'+str(sfid)+'_'+cams+'.pdf'
    if not os.path.lexists('images/'+snapkey):
        os.mkdir('images/'+snapkey)
    f1 = pyplot.figure(figsize=(12.0,12.0), dpi=600)
    pyplot.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,wspace=0.0,hspace=0.0)


    sfids = merF['mergerinfo'][snapkey]['SubfindID'].value
    this_i = sfids==sfid
    
    grid =  merF['mergerinfo'][snapkey]['Tree_SFID_grid'].value[this_i][0]
    
    grid_snaps= merF['mergerinfo'][snapkey]['SnapNums'].value
    print(grid_snaps, grid)
    
    nx=12
    ny=12
    i=0
    j=0
    for afk in all_fil_keys:
        j=j+1
        
        for ask in all_snap_keys:
            i=i+1
            axi=f1.add_subplot(nx,ny,j*12-(i-1)%12,facecolor='Black')
            axi.set_xticks([]) ; axi.set_yticks([])

            imfki= np.asarray(im_rf_fil_keys)==afk
            imfkihst=np.asarray(im_rf_fil_keys_hst)==afk
            
            imski= np.asarray(im_snap_keys)==ask

            #print(imfki,imski,imski[imfki])
            
            if np.sum(imski[imfki])==1.0:
                spinec='Orange'
                zo=99
                spinelw=2
            elif np.sum(imski[imfkihst])==1.0:
                spinec='Gray'
                zo=2
                spinelw=2                
            else:
                spinec='Black'
                zo=1
                spinelw=0.5

            axi.set_zorder(zo)
            
            for ss in axi.spines:
                s=axi.spines[ss]
                s.set_color(spinec)
                s.set_linewidth(spinelw)
                
            
            snap_int=np.int32(ask[-3:])
            search_sfid=grid[grid_snaps==snap_int][0]
            this_z=gsu.redshift_from_snapshot(snap_int)
            
            #subfind IDs in image sample at this snaphshot
            sfid_image_sample = np.asarray(get_all_snap_val(msF,ask,'SubfindID',camera=0))

            in_sample = sfid_image_sample==search_sfid
            print(ask, snap_int, search_sfid, np.sum(in_sample) )
            if np.sum(in_sample)==1:
                #axi,snapkey,subfindID,camera,filters=['NC-F115W','NC-F150W','NC-F200W'],alph=0.2,Q=8.0,Npix=400,sb='SB25',rfkey=None,dx=0,dy=0
                showgalaxy.showgalaxy(axi,ask,search_sfid,cams,filters=[afk,afk,afk],alph=alph,Q=Q,Npix=None,ckpcz=75.0,sb='SB25',rfkey=None)

            if j==1:
                #display redshift at top
                if i==nx:
                    zs='z='
                else:
                    zs=''
                axi.annotate(zs+'{:3.1f}'.format(this_z),(0.5,0.85),xycoords='axes fraction',color='White',ha='center',va='center',fontsize=14)
            if i%nx==0:
                anfk=afk.split('-')[-1]
                anik=afk.split('-')[0]
                if anik=='ACS':
                    anc='Blue'
                if anik=='WFC3':
                    anc='White'
                if anik=='NC':
                    anc='Red'
                axi.annotate(anfk,(0.5,0.15),xycoords='axes fraction',color=anc,ha='center',va='center',fontsize=12,backgroundcolor='Black')
    
    f1.savefig(filen,dpi=600)
    pyplot.close(f1)

    return 1



<<<<<<< HEAD
def do_rf_result_grid(snap_keys_par,fil_keys_par,rflabel='paramsmod',rf_labelfunc='label_merger_forward250_both'):
=======
def do_rf_result_grid(rflabel='paramsmod',rf_labelfunc='label_merger_forward250_both'):
>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
    i=0

    if rflabel=='paramsmod':
        cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc']
    else:
        assert(False)
        
<<<<<<< HEAD
    for sk_rfo,fk_rfo,dkz1,dkz2 in zip(snap_keys_par,fil_keys_par,data_z1_keys,data_z2_keys):
=======
    for sk_rfo,fk_rfo in zip(snap_keys,fil_keys):
>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
        i=i+1

        rfdata = 'rfoutput/'+rf_labelfunc+'/'
        
        data_file = rfdata+rflabel+'_traindata_{}_{}.pkl'.format(sk_rfo,fk_rfo)
        result_file= rfdata+rflabel+'_result_{}_{}.pkl'.format(sk_rfo,fk_rfo)
        labels_file = rfdata+rflabel+'_labels_{}_{}.pkl'.format(sk_rfo,fk_rfo)
        prob_file = rfdata+rflabel+'_label_probability_{}_{}.pkl'.format(sk_rfo,fk_rfo)
        pcs_file= rfdata+rflabel+'_pc_{}_{}.pkl'.format(sk_rfo,fk_rfo)
        roc_file= rfdata+rflabel+'_roc_{}_{}.npy'.format(sk_rfo,fk_rfo)
        obj_file= rfdata+rflabel+'_rfobj_{}_{}.npy'.format(sk_rfo,fk_rfo)
        
        snap_int=np.int32(sk_rfo[-3:])
        rfo_z=gsu.redshift_from_snapshot(snap_int)
        
        if rfo_z > 4.2:
            continue

<<<<<<< HEAD
        rf_objs=np.load(obj_file) #load RF classifier for snap in top loop

=======
        
>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
        for sk_app,fk_app in zip(snap_keys,fil_keys):

            app_snap_int=np.int32(sk_app[-3:])
            app_z = gsu.redshift_from_snapshot(app_snap_int)

            if app_z > 4.2:
                continue
            
            rf_tpr=[]
            rf_fpr=[]
<<<<<<< HEAD
=======
            rf_objs=np.load(obj_file) #load RF classifier for snap in top loop
>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
 
            data_file_app = rfdata+rflabel+'_data_{}_{}.pkl'.format(sk_app,fk_app)
            
            df=np.load(data_file_app)
            rf_flag_app = df['mergerFlag']


            
            #apply these classifiers to the application snapshot data frame
            for rfo in rf_objs:
                prob_i = rfo.clrf.predict_proba(df[cols].values)
<<<<<<< HEAD
                pred_i = prob_i[:,1] > 0.3  #??
=======
                pred_i = prob_i[:,1] > 0.4  #??
>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
                 
                rf_tpr.append(np.sum(rf_flag_app[pred_i])/np.sum(rf_flag_app))
                rf_fpr.append( np.sum(np.logical_and(np.asarray(pred_i)==True,np.asarray(rf_flag_app)==False) )/np.sum( np.asarray(rf_flag_app)==False) )
                
<<<<<<< HEAD
            print('RF (z={:4.2f}) applied to simulation at z={:4.2f}.  '.format(rfo_z,app_z), np.mean(np.asarray(rf_tpr)), np.mean(np.asarray(rf_fpr)) )

        df1,df2,df3=load_candels_dfs(zrange=[dkz1,dkz2])
        datacols=['dGM20','fGM20','ASYM','MPRIME','I','D','CON']
        rf_data=[]
        if rfo_z <= 1.2:
            df_use=df1
        elif rfo_z <=1.7:
            df_use=df2
        else:
            df_use=df3
            
        for rfo in rf_objs:
            prob=rfo.clrf.predict_proba(df_use[datacols].values)
            pred = prob[:,1] > 0.3
            rf_data.append(np.sum(pred))
        print('RF (z={:4.2f}) applied to data at {:4.2f} < z < {:4.2f}.  '.format(rfo_z,dkz1,dkz2), np.mean(rf_data), df2.shape[0])

        
        
=======
            print('RF (z={:4.2f}) applied to data at z={:4.2f}.  '.format(rfo_z,app_z), np.mean(np.asarray(rf_tpr)), np.mean(np.asarray(rf_fpr)) )

    
>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
    return

if __name__=="__main__":


    #morph_stat_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB25_12filters_all_NEW.hdf5'
    #morph_stat_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB25_12filters_all_FILES.hdf5'
    #morph_stat_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB27_12filters_all_NEW.hdf5'
    morph_stat_file = os.path.expandvars('$HOME/Dropbox/Workspace/Papers_towrite/Illustris_ObservingMergers/CATALOGS/nonparmorphs_SB25_12filters_all_FILES.hdf5')



    #merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo.hdf5'
    #merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo_SB27.hdf5'
    #merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo_SB25.hdf5'
    merger_file = os.path.expandvars('$HOME/Dropbox/Workspace/Papers_towrite/Illustris_ObservingMergers/CATALOGS/imagedata_mergerinfo_SB25_2017May08.hdf5')
    merger_file_500 = os.path.expandvars('$HOME/Dropbox/Workspace/Papers_towrite/Illustris_ObservingMergers/CATALOGS/imagedata_mergerinfo_SB25_2017March3.hdf5')


    with h5py.File(morph_stat_file,'r') as msF:
        with h5py.File(merger_file,'r') as merF:
            with h5py.File(merger_file_500,'r') as merF500:
                #localvars = make_sfr_radius_mass_plots(msF,merF,rfiter=10)
                
                #localvars = make_morphology_plots(msF,merF)
                #res = make_pc1_images(msF,merF,bd='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/')
                
                
                rflabel='paramsmod'
<<<<<<< HEAD
                localvars = run_random_forest(msF,merF,snap_keys,fil_keys_hst,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys_hst),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',twin=0.5)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_window500_both',rflabel=rflabel)

                
                localvars = run_random_forest(msF,merF,snap_keys,fil_keys_hst,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_forward250_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys_hst),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_forward250_both',twin=0.25)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_forward250_both',rflabel=rflabel)
                
                
                '''
                localvars = run_random_forest(msF,merF,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_past250_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_past250_both',twin=0.25)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_past250_both',rflabel=rflabel)
                '''
                
=======
                localvars = run_random_forest(msF,merF,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',twin=0.5)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_window500_both',rflabel=rflabel)

                
                localvars = run_random_forest(msF,merF,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_forward250_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_forward250_both',twin=0.25)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_forward250_both',rflabel=rflabel)

                localvars = run_random_forest(msF,merF,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_past250_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_past250_both',twin=0.25)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_past250_both',rflabel=rflabel)

>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
                #don't have merF125 that we need for this one
                #localvars = run_random_forest(msF,merF,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_window250_both',balancetrain=False)
                #localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_window250_both',twin=0.25)
                #res = make_all_structures(msF,merF,rf_labelfunc='label_merger_window250_both',rflabel=rflabel)
<<<<<<< HEAD

                '''
                localvars = run_random_forest(msF,merF,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_past500_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_past500_both',twin=0.5)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_past500_both',rflabel=rflabel)

=======

                localvars = run_random_forest(msF,merF,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_past500_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_past500_both',twin=0.5)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_past500_both',rflabel=rflabel)

>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
                
                #base forward-500 runs on old file merF500!
                localvars = run_random_forest(msF,merF500,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_forward500_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_forward500_both',twin=0.5)
                res = make_all_structures(msF,merF500,rf_labelfunc='label_merger_forward500_both',rflabel=rflabel)
<<<<<<< HEAD
                '''
=======
                
>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
                

                #res = make_merger_images(msF,merF,rflabel='params',rf_masscut=10.0**(10.5),labelfunc='label_merger4')
                
                
                
                #res = make_morphology_evolution_plots(msF,merF)
                

                '''
                sfid_035 = msF['nonparmorphs']['snapshot_035']['SubfindID'].value
                
                print(sfid_035)

                for sf in sfid_035:
                    res=make_filter_images(msF,merF,sfid=sf,snapkey='snapshot_035',alph=0.5,Q=5.0)
                '''

                
                #res=make_filter_images(msF,merF,sfid=202,cams='00',snapkey='snapshot_035',alph=0.5,Q=5.0)
                #res=make_filter_images(msF,merF,sfid=202,cams='01',snapkey='snapshot_035',alph=0.5,Q=5.0)
                #res=make_filter_images(msF,merF,sfid=202,cams='02',snapkey='snapshot_035',alph=0.5,Q=5.0)
                #res=make_filter_images(msF,merF,sfid=202,cams='03',snapkey='snapshot_035',alph=0.5,Q=5.0)
                
                    
                '''
                res=make_filter_images(msF,merF,sfid=106,snapkey='snapshot_038',alph=0.5,Q=5.0)
                res=make_filter_images(msF,merF,sfid=1274,snapkey='snapshot_038',alph=0.5,Q=5.0)
                res=make_filter_images(msF,merF,sfid=1325,snapkey='snapshot_038',alph=0.5,Q=5.0)
                res=make_filter_images(msF,merF,sfid=139,snapkey='snapshot_038',alph=0.5,Q=5.0)
                res=make_filter_images(msF,merF,sfid=263,snapkey='snapshot_038',alph=0.5,Q=5.0)
                res=make_filter_images(msF,merF,sfid=289,snapkey='snapshot_038',alph=0.5,Q=5.0)
                res=make_filter_images(msF,merF,sfid=432,snapkey='snapshot_038',alph=0.5,Q=5.0)
                '''

                '''
                sfid=76287
                masshistory.masshistory('snapshot_068',sfid,camnum=0,size=6,alph=0.5,
                                        Q=5,sb='SB25',gyrbox=False,ckpc=50.0,Npix=None,
                                        use_rfkey=True,trange=[-0.71,0.71],plot_rfs=True,savefile='test00.pdf')
                masshistory.masshistory('snapshot_068',sfid,camnum=1,size=6,alph=0.5,
                                        Q=5,sb='SB25',gyrbox=False,ckpc=50.0,Npix=None,
                                        use_rfkey=True,trange=[-0.71,0.71],plot_rfs=True,savefile='test01.pdf')
                masshistory.masshistory('snapshot_068',sfid,camnum=2,size=6,alph=0.5,
                                        Q=5,sb='SB25',gyrbox=False,ckpc=50.0,Npix=None,
                                        use_rfkey=True,trange=[-0.71,0.71],plot_rfs=True,savefile='test02.pdf')                
                masshistory.masshistory('snapshot_068',sfid,camnum=3,size=6,alph=0.5,
                                        Q=5,sb='SB25',gyrbox=False,ckpc=50.0,Npix=None,
                                        use_rfkey=True,trange=[-0.71,0.71],plot_rfs=True,savefile='test03.pdf')
                '''

                
                #
                #res=print_table_one(msF,merF)

                

<<<<<<< HEAD
                res=do_rf_result_grid(copy.copy(snap_keys),copy.copy(fil_keys_hst),rflabel='paramsmod',rf_labelfunc='label_merger_window500_both')
=======
                #res=do_rf_result_grid(rflabel='paramsmod',rf_labelfunc='label_merger_window500_both')
>>>>>>> 640c21f221b0b6f8240adb943d66e69cedf8a76a
