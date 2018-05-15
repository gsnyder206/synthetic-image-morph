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
import numpy.random as nprand
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
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import merge_field

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

im_rf_fil_keys = ['ACS-F435W','ACS-F606W','ACS-F606W','NC-F115W',
                  'NC-F115W','NC-F115W','NC-F115W','NC-F200W']

im_rf_fil_keys2 = ['ACS-F606W','ACS-F814W','NC-F115W','NC-F150W',
                  'NC-F150W','NC-F200W','NC-F200W','NC-F277W']

im_rf_fil_keys_hst= ['ACS-F814W','ACS-F814W','ACS-F814W','ACS-F814W','ACS-F814W','ACS-F814W','ACS-F814W','ACS-F814W',]


im_rf_fil_keys_hst2= ['WFC3-F160W','WFC3-F160W','WFC3-F160W','WFC3-F160W',
                  'WFC3-F160W','WFC3-F160W','WFC3-F160W','WFC3-F160W']


snap_keys = im_snap_keys
fil_keys = im_rf_fil_keys
fil_keys_hst=im_rf_fil_keys_hst
fil_keys_hst2=im_rf_fil_keys_hst2


data_z1_keys=[0.25,0.75,1.25,1.75,2.25,2.75,3.5,4.5]
data_z2_keys=[0.75,1.25,1.75,2.25,2.75,3.5,4.5,5.5]

data_fil_keys=['I','I','J','J','H','H','H','H']

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



class simple_forest():
    def __init__(self,rfc):
        self.rfo=rfc
        return


def do_masshistory(sk,sfid,cam,alph=0.2,Q=5.0,ckpc=50.0,lab='',subdir=None,relpath=None,interact=False,**kwargs):
    if interact is True:
        fn=None
    else:
        if subdir is None:
            fn1='masshistory/'+lab+'_masshistory_'+sk+'_sh{:}_cam{:}.svg'.format(sfid,cam)#multi merger
        else:
            fn1='masshistory/'+subdir+'/'+lab+'_masshistory_'+sk+'_sh{:}_cam{:}.pdf'.format(sfid,cam)#multi merger

        if relpath is None:
            fn=fn1
        else:
            fn=os.path.join(relpath,fn1)
            
    masshistory.masshistory(sk,sfid,camnum=cam,size=6.0,alph=alph,
                            Q=Q,sb='SB25',gyrbox=False,ckpc=ckpc,Npix=None,
                            trange=[-0.71,1.01],savefile=fn,**kwargs)  
    return fn1


def plot_mass_history_all(sk,tp_id,tp_cam,fp_id,fp_cam,tn_id,tn_cam,fn_id,fn_cam):

    alph=0.2 ; Q=6.0
    if not os.path.lexists('masshistory'):
        os.mkdir('masshistory')
    if not os.path.lexists('masshistory/'+sk):
        os.mkdir('masshistory/'+sk)

    for idn,cam in zip(tp_id,tp_cam):
        do_masshistory(sk,idn,int(cam),alph=alph,Q=Q,lab='tp',subdir=sk)
    for idn,cam in zip(fp_id,fp_cam):
        do_masshistory(sk,idn,int(cam),alph=alph,Q=Q,lab='fp',subdir=sk)
    for idn,cam in zip(tn_id,tn_cam):
        do_masshistory(sk,idn,int(cam),alph=alph,Q=Q,lab='tn',subdir=sk)
    for idn,cam in zip(fn_id,fn_cam):
        do_masshistory(sk,idn,int(cam),alph=alph,Q=Q,lab='fn',subdir=sk)
        
    return



def do_mass_history_examples():


    alph=0.2 ; Q=6.0
    cam=0
    if not os.path.lexists('masshistory'):
        os.mkdir('masshistory')
    

    sk='snapshot_075'
    do_masshistory(sk,61699,3,alph=alph,Q=Q,lab='tp')
    do_masshistory(sk,97315,0,alph=alph,Q=Q,lab='tp')
    do_masshistory(sk,37227,2,alph=alph,Q=Q,lab='tp')
    do_masshistory(sk,64667,0,alph=alph,Q=Q,lab='tp')

    do_masshistory(sk,17217,1,alph=alph,Q=Q,lab='fp')
    do_masshistory(sk,104570,2,alph=alph,Q=Q,lab='fp')
    do_masshistory(sk,76157,3,alph=alph,Q=Q,lab='fp')
    do_masshistory(sk,38239,1,alph=alph,Q=Q,lab='fp')
    
    
    do_masshistory(sk,198973,2,alph=alph,Q=Q,lab='tn')
    do_masshistory(sk,24679,2,alph=alph,Q=Q,lab='tn')
    do_masshistory(sk,145934,0,alph=alph,Q=Q,lab='tn')
    do_masshistory(sk,27042,0,alph=alph,Q=Q,lab='tn')


    
    do_masshistory(sk,105660,3,alph=alph,Q=Q,lab='fn')
    do_masshistory(sk,76609,1,alph=alph,Q=Q,lab='fn')
    do_masshistory(sk,64667,1,alph=alph,Q=Q,lab='fn')
    do_masshistory(sk,64667,3,alph=alph,Q=Q,lab='fn')

    do_masshistory(sk,97125,0,alph=alph,Q=Q,lab='fn')
    do_masshistory(sk,138402,3,alph=alph,Q=Q,lab='fn')
    do_masshistory(sk,17216,0,alph=alph,Q=Q,lab='fn')

    


    return
    


def simple_random_forest(msF,merF,snap_key,fil_key,fil2_key=None,paramsetting='medium',rfiter=3,rf_masscut=10.0**(10.5),labelfunc='label_merger1',
                         n_estimators=2000,max_leaf_nodes=-1,max_features=4,balancetrain=True,skip_mi=False,trainfrac=0.67, traininglabel='mergerFlag',skipcalc=False,seed=0,**kwargs):

    sk=snap_key
    fk=fil_key



    
    
    asym = get_all_morph_val(msF,sk,fk,'ASYM')
    gini = get_all_morph_val(msF,sk,fk,'GINI')
    m20 = get_all_morph_val(msF,sk,fk,'M20')
    cc = get_all_morph_val(msF,sk,fk,'CC')
    Mstat = get_all_morph_val(msF,sk,fk,'MID2_MPRIME')
    Istat = get_all_morph_val(msF,sk,fk,'MID2_ISTAT')
    Dstat = get_all_morph_val(msF,sk,fk,'MID2_DSTAT')
    rpet=get_all_morph_val(msF,sk,fk,'RP')
    pix_arcsec=get_all_morph_val(msF,sk,fk,'PIX_ARCSEC')

    redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
    assert redshift > 0.0
    scale=illcos.kpc_proper_per_arcmin(redshift).value/60.0
    
    size = np.log10(rpet*pix_arcsec*scale)


    
    snpix= get_all_morph_val(msF,sk,fk,'SNPIX')
    if fil2_key is not None:
        fk2=fil2_key
        asym_2 = get_all_morph_val(msF,sk,fk2,'ASYM')
        gini_2 = get_all_morph_val(msF,sk,fk2,'GINI')
        m20_2 = get_all_morph_val(msF,sk,fk2,'M20')
        cc_2 = get_all_morph_val(msF,sk,fk2,'CC')
        Mstat_2 = get_all_morph_val(msF,sk,fk2,'MID2_MPRIME')
        Istat_2 = get_all_morph_val(msF,sk,fk2,'MID2_ISTAT')
        Dstat_2 = get_all_morph_val(msF,sk,fk2,'MID2_DSTAT')
        rpet_2 = get_all_morph_val(msF,sk,fk2,'RP')
        pix_arcsec2=get_all_morph_val(msF,sk,fk2,'PIX_ARCSEC')

        size_2 = np.log10(rpet_2*pix_arcsec2*scale)
        
        snpix_2 = get_all_morph_val(msF,sk,fk2,'SNPIX')
        max_features=3
        
        if paramsetting=='twofilters_snp':
            cols=['dGM20','fGM20','asym','Dstat','cc','snp','dGM20_2','fGM20_2','asym_2','Dstat_2','cc_2','snp_2']
            rflabel='twofilters_snp'
            plotlabels=np.asarray([r'$GMS_{'+fk[-5:]+'}$',
                                   r'$GMF_{'+fk[-5:]+'}$',
                                   r'$A_{'+fk[-5:]+'}$',
                                   r'$D_{'+fk[-5:]+'}$',
                                   r'$C_{'+fk[-5:]+'}$',
                                   r'SN/pix$_{'+fk[-5:]+'}$',
                                   r'$GMS_{'+fk2[-5:]+'}$',
                                   r'$GMF_{'+fk2[-5:]+'}$',
                                   r'$A_{'+fk2[-5:]+'}$',
                                   r'$D_{'+fk2[-5:]+'}$',
                                   r'$C_{'+fk2[-5:]+'}$',
                                   r'SN/pix$_{'+fk2[-5:]+'}$'])
        elif paramsetting=='twofilters_snp_mi':
            cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc','snp','dGM20_2','fGM20_2','asym_2','Mstat_2','Istat_2','Dstat_2','cc_2','snp_2']
            rflabel='twofilters_snp_mi'
            max_features=3
            plotlabels=np.asarray([r'$GMS_{'+fk[-5:]+'}$',
                                   r'$GMF_{'+fk[-5:]+'}$',
                                   r'$A_{'+fk[-5:]+'}$',
                                   r'$M_{'+fk[-5:]+'}$',
                                   r'$I_{'+fk[-5:]+'}$',
                                   r'$D_{'+fk[-5:]+'}$',
                                   r'$C_{'+fk[-5:]+'}$',
                                   r'SN/pix$_{'+fk[-5:]+'}$',
                                   r'$GMS_{'+fk2[-5:]+'}$',
                                   r'$GMF_{'+fk2[-5:]+'}$',
                                   r'$A_{'+fk2[-5:]+'}$',
                                   r'$M_{'+fk2[-5:]+'}$',
                                   r'$I_{'+fk2[-5:]+'}$',
                                   r'$D_{'+fk2[-5:]+'}$',
                                   r'$C_{'+fk2[-5:]+'}$',
                                   r'SN/pix$_{'+fk2[-5:]+'}$'])            
        else:
            cols=['dGM20','fGM20','asym','Dstat','cc','dGM20_2','fGM20_2','asym_2','Dstat_2','cc_2']
            rflabel='twofilters'
            plotlabels=np.asarray([r'$GMS_{'+fk[-5:]+'}$',
                                   r'$GMF_{'+fk[-5:]+'}$',
                                   r'$A_{'+fk[-5:]+'}$',
                                   r'$D_{'+fk[-5:]+'}$',
                                   r'$C_{'+fk[-5:]+'}$',
                                   r'$GMS_{'+fk2[-5:]+'}$',
                                   r'$GMF_{'+fk2[-5:]+'}$',
                                   r'$A_{'+fk2[-5:]+'}$',
                                   r'$D_{'+fk2[-5:]+'}$',
                                   r'$C_{'+fk2[-5:]+'}$'])            



        

        
    elif paramsetting=='medium':
        cols=['dGM20','fGM20','asym','Dstat','cc','logssfr','size','log_mstar_mhalo','bhmdot']
        rflabel='simpletest'
        max_features=3

    elif paramsetting=='max':
        cols=['dGM20','fGM20','asym','Dstat','cc','logssfr','size','log_mstar_mhalo','bhmdot','elong','flag','m_a','m_i2','mid1_gini','mid1_m20']        
        rflabel='maxtest'
        max_features=4
    elif paramsetting=='shapeonly':
        cols=['dGM20','fGM20','asym','Dstat','cc','elong','m_a','m_i2','rp','mid1_gini','mid1_m20']        
        rflabel='shapetest'
        max_features=4        
    elif paramsetting=='onefilter':
        cols=['dGM20','fGM20','asym','Dstat','cc']  #M, I, SNP? 
        rflabel='onefilter'
        max_features=3
        plotlabels=np.asarray([r'$GMS_{'+fk[-5:]+'}$',
                               r'$GMF_{'+fk[-5:]+'}$',
                               r'$A_{'+fk[-5:]+'}$',
                               r'$D_{'+fk[-5:]+'}$',
                               r'$C_{'+fk[-5:]+'}$'])  #,
                               #r'SN/pix$_{'+fk[-5:]+'}$']
                               
    elif paramsetting=='onefilter_snp':
        cols=['dGM20','fGM20','asym','Dstat','cc','snp']
        rflabel='onefilter_snp'
        max_features=3
        plotlabels=np.asarray([r'$GMS_{'+fk[-5:]+'}$',
                               r'$GMF_{'+fk[-5:]+'}$',
                               r'$A_{'+fk[-5:]+'}$',
                               r'$D_{'+fk[-5:]+'}$',
                               r'$C_{'+fk[-5:]+'}$',
                               r'SN/pix$_{'+fk[-5:]+'}$'])
    
                               
    sfid = get_all_snap_val(msF,sk,'SubfindID')
        
    boolean_flag,number,t_lastMaj,t_lastMin,t_nextMaj,t_nextMin = eval(labelfunc+'(merF,sk)')

    mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
    mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
    log_mstar_mhalo = np.log10( mstar/mhalo )
    sfr = get_all_snap_val(msF,sk,'SFR_Msunperyr')
    bhmdot = get_all_snap_val(msF,sk,'BHMdot_Msunperyr')
    


    fk_imfile = np.asarray(get_all_morph_val(msF,sk,fk,'IMFILES'),dtype='str')
    if fil2_key is not None:
        f2k_imfile = np.asarray(get_all_morph_val(msF,sk,fil2_key,'IMFILES'),dtype='str')

    
    rf_dict={}

    
    if fil2_key is not None:
        gi = np.where(np.isfinite(gini)*np.isfinite(m20)*np.isfinite(asym)*np.isfinite(Dstat)*np.isfinite(cc)*(snpix >= 3.0)*(snpix_2 >= 3.0)*
                      np.isfinite(gini_2)*np.isfinite(m20_2)*np.isfinite(asym_2)*np.isfinite(Dstat_2)*np.isfinite(cc_2)*(mstar >= rf_masscut) != 0)[0]
        S_GM20 = SGM20(gini[gi],m20[gi])
        F_GM20 = FGM20(gini[gi],m20[gi])
        
        rf_dict['dGM20']=S_GM20
        rf_dict['fGM20']=F_GM20
        rf_dict['asym']=asym[gi]
        rf_dict['Mstat']=np.log10(1.0+Mstat[gi])
        rf_dict['Istat']=np.log10(1.0+Istat[gi])
        rf_dict['size']=size[gi] #already log kpc
        rf_dict['Dstat']=np.log10(1.0+Dstat[gi])   #NOTE log quantity in paramsmod2
        rf_dict['cc']=cc[gi]

        S_GM20_2 = SGM20(gini_2[gi],m20_2[gi])
        F_GM20_2 = FGM20(gini_2[gi],m20_2[gi])
        
        rf_dict['dGM20_2']=S_GM20_2
        rf_dict['fGM20_2']=F_GM20_2
        rf_dict['asym_2']=asym_2[gi]
        rf_dict['Mstat_2']=np.log10(1.0+Mstat_2[gi])
        rf_dict['Istat_2']=np.log10(1.0+Istat_2[gi])
        rf_dict['size_2']=size_2[gi] #already log kpc
        rf_dict['Dstat_2']=np.log10(1.0+Dstat_2[gi])   #NOTE log quantity in paramsmod2
        rf_dict['cc_2']=cc_2[gi]
        rf_dict['snp_2']=snpix_2[gi]

        rf_dict['IM2FILE']=f2k_imfile[gi]
        
    else:
        gi = np.where(np.isfinite(gini)*np.isfinite(m20)*np.isfinite(asym)*np.isfinite(Dstat)*np.isfinite(cc)*(snpix >= 3.0)*(mstar >= rf_masscut) != 0)[0]

        S_GM20 = SGM20(gini[gi],m20[gi])
        F_GM20 = FGM20(gini[gi],m20[gi])
        
        rf_dict['dGM20']=S_GM20
        rf_dict['fGM20']=F_GM20
        rf_dict['asym']=asym[gi]
        rf_dict['Mstat']=Mstat[gi]
        rf_dict['Istat']=Istat[gi]
        rf_dict['size']=size[gi] #already log kpc
        rf_dict['Dstat']=np.log10(Dstat[gi])   #NOTE log quantity in paramsmod2
        rf_dict['cc']=cc[gi]



    rf_dict['IMFILE']=fk_imfile[gi]
    
    rf_dict['elong']= get_all_morph_val(msF,sk,fk,'ELONG')[gi]
    rf_dict['flag']= get_all_morph_val(msF,sk,fk,'FLAG')[gi]
    rf_dict['m_a']= get_all_morph_val(msF,sk,fk,'M_A')[gi]
    rf_dict['m_i2']= get_all_morph_val(msF,sk,fk,'M_I2')[gi]
    #rf_dict['rp']= get_all_morph_val(msF,sk,fk,'RP')[gi]
    rf_dict['mid1_gini']= get_all_morph_val(msF,sk,fk,'MID1_GINI')[gi]
    rf_dict['mid1_m20']= get_all_morph_val(msF,sk,fk,'MID1_M20')[gi]
    
    rf_dict['mergerFlag']=boolean_flag[gi]
    rf_dict['SubfindID']=sfid[gi]
    rf_dict['Mstar_Msun']=mstar[gi]
    rf_dict['SFR_Msunperyr']=sfr[gi]
    rf_dict['mergerNumber']=number[gi]

    rf_dict['snp']=snpix[gi]

    sig=0.0
    rf_dict['logssfr'] = np.log10( np.float64(sfr[gi])+1.0e-5) - np.log10(np.float64(mstar[gi])) + sig*nprand.randn(gi.shape[0])
    rf_dict['mhalo']=np.log10(mhalo[gi]) + sig*nprand.randn(gi.shape[0])
    rf_dict['log_mstar_mhalo']=log_mstar_mhalo[gi] + sig*nprand.randn(gi.shape[0])
    rf_dict['bhmdot']=np.log10(bhmdot[gi]+1.0e-5) + sig*nprand.randn(gi.shape[0])


    
    rf_dict['gini']=gini[gi]
    rf_dict['m20']=m20[gi]

    rf_dict['t_lastMaj']=t_lastMaj[gi]
    rf_dict['t_lastMin']=t_lastMin[gi]
    rf_dict['t_nextMaj']=t_nextMaj[gi]
    rf_dict['t_nextMin']=t_nextMin[gi]


    if max_leaf_nodes==-1:
        max_leaf_nodes= np.int64( 0.5*np.sum(rf_dict['mergerFlag']) )

    
    df=pandas.DataFrame(rf_dict)


    
    trainDF, testDF = PyML.trainingSet(df,training_fraction=trainfrac,seed=seed)

    #create cross-validation test
    train = trainDF[cols].values
    test = testDF[cols].values
    labels = trainDF[traininglabel]


    if skipcalc==True:        
        return rflabel,cols,plotlabels,0.0
    
    rfc = RandomForestClassifier(n_jobs=-1,oob_score=True,n_estimators=n_estimators,max_leaf_nodes=max_leaf_nodes,
                                 max_features=max_features)  #,class_weight='balanced_subsample'


    rfc.fit(train,labels) 
            
    test_preds = np.asarray(rfc.predict(test))
    test_probs = np.asarray(rfc.predict_proba(test))
    test_feature_importances = rfc.feature_importances_


    
    all_preds = np.asarray(rfc.predict(df[cols].values))
    all_probs = np.asarray(rfc.predict_proba(df[cols].values))

    train_preds = np.asarray(rfc.predict(trainDF[cols].values))
    train_probs = np.asarray(rfc.predict_proba(trainDF[cols].values))
    
    Nroc=100
    threshes= np.logspace(-2.0,0,Nroc)
    ROCstats = {'thresh':np.zeros(Nroc), 'tpr':np.zeros(Nroc),'fpr':np.zeros(Nroc),'ppv':np.zeros(Nroc),'mcc':np.zeros(Nroc)}
    ROCtests = {'thresh':np.zeros(Nroc), 'tpr':np.zeros(Nroc),'fpr':np.zeros(Nroc),'ppv':np.zeros(Nroc),'mcc':np.zeros(Nroc)}
    ROCtrain = {'thresh':np.zeros(Nroc), 'tpr':np.zeros(Nroc),'fpr':np.zeros(Nroc),'ppv':np.zeros(Nroc),'mcc':np.zeros(Nroc)}

    for j,tv in enumerate(threshes):
        ROC_results=PyML.confusionMatrix(df,all_probs,threshold=tv,traininglabel=traininglabel)  #compute ROC curve
        ROC_tests=PyML.confusionMatrix(testDF,test_probs,threshold=tv,traininglabel=traininglabel)
        ROC_train=PyML.confusionMatrix(trainDF,train_probs,threshold=tv,traininglabel=traininglabel)

        for rs in ROCstats.keys():
            ROCstats[rs][j]=ROC_results[rs]
            ROCtests[rs][j]=ROC_tests[rs]
            ROCtrain[rs][j]=ROC_train[rs]
    

    athreshes=np.linspace(-0.3,1.2,Nroc)
    atpr=np.zeros_like(athreshes)
    afpr=np.zeros_like(athreshes)
    appv=np.zeros_like(athreshes)

    for i,at in enumerate(athreshes):
        asym_classifier = testDF['asym'].values > at
        asym_com,asym_ppv,asym_tpr,asym_fpr = simple_classifier_stats(asym_classifier,testDF[traininglabel].values)
        atpr[i]=asym_tpr
        afpr[i]=asym_fpr
        appv[i]=asym_ppv

    dthreshes=np.linspace(-0.3,0.7,Nroc)
    dtpr=np.zeros_like(dthreshes)
    dfpr=np.zeros_like(dthreshes)
    dppv=np.zeros_like(dthreshes)

    for i,dt in enumerate(dthreshes):
        dg_classifier = testDF['dGM20'].values > dt
        dg_com,dg_ppv,dg_tpr,dg_fpr = simple_classifier_stats(dg_classifier,testDF[traininglabel].values)
        dtpr[i]=dg_tpr
        dfpr[i]=dg_fpr
        dppv[i]=dg_ppv


    tpr_minus_fpr = ROCtests['tpr']-ROCtests['fpr']

    balance_point = np.abs( 1.0-ROCtests['tpr']-ROCtests['fpr'])
    
    frac_error_thing = np.where( ROCtests['ppv']*ROCtests['tpr'] > 0.0 , (ROCtests['ppv']+ROCtests['tpr'])/(ROCtests['ppv']*ROCtests['tpr']), 1e6*np.ones_like(threshes) )
    
    best_i1 = np.argmax(tpr_minus_fpr)
    best_tpr1=ROCtests['tpr'][best_i1]
    best_fpr1=ROCtests['fpr'][best_i1]
    best_th1 =ROCtests['thresh'][best_i1]
    
    best_i2 = np.argmin(frac_error_thing)
    best_tpr2=ROCtests['tpr'][best_i2]
    best_fpr2=ROCtests['fpr'][best_i2]
    best_th2 =ROCtests['thresh'][best_i2]

    best_i3 = np.argmin(balance_point)
    best_tpr3=ROCtests['tpr'][best_i3]
    best_fpr3=ROCtests['fpr'][best_i3]
    best_th3 =ROCtests['thresh'][best_i3]

    best_i4 = np.argmax(ROCtests['mcc'])
    best_tpr4=ROCtests['tpr'][best_i4]
    best_fpr4=ROCtests['fpr'][best_i4]
    best_th4 =ROCtests['thresh'][best_i4]    

    best_th=best_th3

    
    ROCtests['frac_error']=frac_error_thing
    ROCtests['balance_point']=balance_point
    ROCtests['best_i_frac_error']=best_i2
    ROCtests['best_i_balance_point']=best_i3
    ROCtests['best_i_mcc']=best_i4

    ROCtests['thresh_color_balance_point']='Orange'
    ROCtests['thresh_color_frac_error']='Magenta'
    ROCtests['thresh_color_mcc']='DodgerBlue'


    if not os.path.lexists('rf_plots'):
        os.mkdir('rf_plots')
    
    if not os.path.lexists('rf_plots/'+labelfunc):
        os.mkdir('rf_plots/'+labelfunc)
    
    roc_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_roc_stats.pdf'
    f3 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)
    axi3=f3.add_subplot(1,1,1)
        
    axi3.plot(ROCstats['fpr'],ROCstats['tpr'],lw=1.0,linestyle='dashed',color='Blue')
    axi3.plot(ROCtests['fpr'],ROCtests['tpr'],lw=2, linestyle='solid',color='Black')
    axi3.plot(ROCtrain['fpr'],ROCtrain['tpr'],lw=1.0,linestyle='dotted',color='Red')

    
    axi3.plot(afpr,atpr,lw=0.5,linestyle='dotted',color='Gray')
    
    
    axi3.plot(dfpr,dtpr,lw=0.5,linestyle='dotted',color='Green')

    axi3.plot( [np.interp(0.25,athreshes,afpr)],[np.interp(0.25,athreshes,atpr)],'o',color='Gray',markersize=4)
    axi3.plot( [np.interp(0.10,dthreshes,dfpr)],[np.interp(0.10,dthreshes,dtpr)],'o',color='Green',markersize=4)

    
    axi3.plot([best_fpr2],[best_tpr2],'o',color=ROCtests['thresh_color_frac_error'],markersize=6,linestyle='None')
    axi3.plot([best_fpr3],[best_tpr3],'s',color=ROCtests['thresh_color_balance_point'],markersize=6,linestyle='None')
    axi3.plot([best_fpr4],[best_tpr4],'^',color=ROCtests['thresh_color_mcc'],markersize=6,linestyle='None')
    
    axi3.legend(['all','cross-val','train','Asymmetry',r'$GMS$'],loc='lower right',fontsize=10)
    
    axi3.set_xlim(-0.05,1.05)
    axi3.set_ylim(-0.05,1.05)
    axi3.set_xlabel('False Positive Rate')
    axi3.set_ylabel('True Positive Rate')
    f3.savefig(roc_filen,dpi=300)
    pyplot.close(f3)




    rocl_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_roc_stats_light.pdf'
    f11 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)
    axi3=f11.add_subplot(1,1,1)
        
    #axi3.plot(ROCstats['fpr'],ROCstats['tpr'],lw=1.0,linestyle='dashed',color='Blue')
    axi3.plot(ROCtests['fpr'],ROCtests['tpr'],lw=2, linestyle='solid',color='Black')
    #axi3.plot(ROCtrain['fpr'],ROCtrain['tpr'],lw=1.0,linestyle='dotted',color='Red')

    
    axi3.plot(afpr,atpr,lw=0.5,linestyle='dotted',color='Gray')
    
    
    axi3.plot(dfpr,dtpr,lw=0.5,linestyle='dotted',color='Green')

    axi3.plot( [np.interp(0.25,athreshes,afpr)],[np.interp(0.25,athreshes,atpr)],'o',color='Gray',markersize=4)
    axi3.plot( [np.interp(0.10,dthreshes,dfpr)],[np.interp(0.10,dthreshes,dtpr)],'o',color='Green',markersize=4)

    
    axi3.plot([best_fpr2],[best_tpr2],'o',color=ROCtests['thresh_color_frac_error'],markersize=6,linestyle='None')
    axi3.plot([best_fpr3],[best_tpr3],'s',color=ROCtests['thresh_color_balance_point'],markersize=6,linestyle='None')
    axi3.plot([best_fpr4],[best_tpr4],'^',color=ROCtests['thresh_color_mcc'],markersize=6,linestyle='None')
    
    axi3.legend(['cross-validation','Asymmetry',r'$GMS$'],loc='lower right',fontsize=10)
    
    axi3.set_xlim(-0.05,1.05)
    axi3.set_ylim(-0.05,1.05)
    axi3.set_xlabel('False Positive Rate')
    axi3.set_ylabel('True Positive Rate')
    f11.savefig(rocl_filen,dpi=300)
    pyplot.close(f11)


    

        
    roc_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_roc_thresh.pdf'
    f3 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.96, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)
    axi3=f3.add_subplot(1,1,1)
        
    #axi3.semilogx(ROCstats['thresh'],ROCstats['tpr'],lw=1,linestyle='solid',color='Blue')
    #axi3.semilogx(ROCstats['thresh'],ROCstats['fpr'],lw=1,linestyle='dashed',color='Blue')
    #axi3.semilogx(ROCstats['thresh'],ROCstats['ppv'],lw=1,linestyle='dotted',color='Blue')

    axi3.semilogx(ROCtests['thresh'],ROCtests['tpr'],lw=2,linestyle='solid',color='Black')
    axi3.semilogx(ROCtests['thresh'],ROCtests['fpr'],lw=2,linestyle='dashed',color='Black')
    axi3.semilogx(ROCtests['thresh'],ROCtests['ppv'],lw=2,linestyle='dotted',color='Black')

    #axi3.semilogx(ROCtrain['thresh'],ROCtrain['tpr'],lw=1,linestyle='solid',color='Red')
    #axi3.semilogx(ROCtrain['thresh'],ROCtrain['fpr'],lw=1,linestyle='dashed',color='Red')
    #axi3.semilogx(ROCtrain['thresh'],ROCtrain['ppv'],lw=1,linestyle='dotted',color='Red')

    factor=(0.2/np.min(frac_error_thing))
    axi3.semilogx(ROCtests['thresh'],frac_error_thing*factor,lw=2,linestyle='solid',color=ROCtests['thresh_color_frac_error'],alpha=0.7)
    axi3.semilogx(ROCtests['thresh'],balance_point,lw=2,linestyle='solid',color=ROCtests['thresh_color_balance_point'],alpha=0.7)
    axi3.semilogx(ROCtests['thresh'],ROCtests['mcc'],lw=2,linestyle='solid',color=ROCtests['thresh_color_mcc'],alpha=0.7)

    #axi3.plot([ROCtests['thresh'][best_i1],ROCtests['thresh'][best_i1]],[-1,2],color='Gray',lw=0.5)
    axi3.legend(['TPR(RF)','FPR(RF)','PPV(RF)','1/F1 Score','TPR=1-FPR','MCC'],fontsize=8,ncol=3,loc='upper center',handletextpad=0,columnspacing=1)


    axi3.plot([ROCtests['thresh'][best_i2]],[frac_error_thing[best_i2]*factor],'o',color=ROCtests['thresh_color_frac_error'],markersize=6,linestyle='None')
    axi3.plot([ROCtests['thresh'][best_i3]],[balance_point[best_i3]],'s',color=ROCtests['thresh_color_balance_point'],markersize=6,linestyle='None')
    axi3.plot([ROCtests['thresh'][best_i4]],[ROCtests['mcc'][best_i4]],'^',color=ROCtests['thresh_color_mcc'],markersize=6,linestyle='None')
    

    
    at_normed=(athreshes-athreshes.min(0))/athreshes.ptp(0)
    dt_normed=(dthreshes-dthreshes.min(0))/dthreshes.ptp(0)

    #these really are a different beast, I think
    #axi3.semilogx(at_normed,atpr,lw=0.25,linestyle='dotted',color='Gray')
    #axi3.semilogx(at_normed,afpr,lw=0.25,linestyle='dashdot',color='Gray')
    #axi3.semilogx(dt_normed,dtpr,lw=0.25,linestyle='dotted',color='Green')
    #axi3.semilogx(dt_normed,dfpr,lw=0.25,linestyle='dashdot',color='Green')
    
    #axi3.legend(['TPR','FPR','TPR(Asym)','FPR(Asym)','TPR(GMS)','FPR(GMS)'],loc='lower left',fontsize=10)
    #axi3.legend(['TPR(all)','FPR(all)','PPV(all)','TPR(cross-val)','FPR(cross-val)','PPV(cross-val)','TPR(train)','FPR(train)','PPV(train)'],loc='lower left',fontsize=8,ncol=3)
    
    axi3.set_ylim(-0.05,1.25)
    axi3.set_xlim(0.015,1.05)
    axi3.set_xlabel('Threshold')
    axi3.set_ylabel('Rate')
    f3.savefig(roc_filen,dpi=300)
    pyplot.close(f3)

    


    test_feature_importances=np.asarray(test_feature_importances)
    cols=np.asarray(cols)
    
    feature_importance = 100.0*(test_feature_importances/test_feature_importances.max())
    sorted_idx=np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5


    
    grid_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_importances.pdf'
    f6 = pyplot.figure(figsize=(5.5,5.5), dpi=300)
    pyplot.subplots_adjust(left=0.20, right=0.99, bottom=0.10, top=0.99,wspace=0.0,hspace=0.0)

    pyplot.barh(pos, feature_importance[sorted_idx], align='center')
    pyplot.yticks(pos, plotlabels[sorted_idx])
    pyplot.xlabel('Relative Importance')

    f6.savefig(grid_filen,dpi=300)
    pyplot.close(f6)     



    
    ncol=len(cols)
    
    rf_probs = all_probs[:,1]
    
    rf_class = rf_probs >= best_th
    rf_flag = df[traininglabel].values

    rf_mass = np.log10( df['Mstar_Msun'].values)

    ssfr_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_ssfr.pdf'
    f7 = pyplot.figure(figsize=(5.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.17, right=0.95, bottom=0.17, top=0.95,wspace=0.0,hspace=0.0)

    
    axi=f7.add_subplot(121)
    axi.locator_params(nbins=5,prune='both')

    m_set  = rf_flag == True
    nm_set = rf_flag == False

    m_set_rf = rf_probs >= best_th
    nm_set_rf = rf_probs < best_th
    
    ssfr_mergers=rf_dict['logssfr'][m_set]
    mass_mergers=np.log10( rf_dict['Mstar_Msun'][m_set])
    ssfr_nonmergers=rf_dict['logssfr'][nm_set]
    mass_nonmergers=np.log10( rf_dict['Mstar_Msun'][nm_set])

    
    sns.kdeplot(mass_mergers,ssfr_mergers,shade=False,shade_lowest=False,cmap='Blues',alpha=0.6,linewidths=1)
    sns.kdeplot(mass_nonmergers,ssfr_nonmergers,shade=False,shade_lowest=False,cmap='Oranges',alpha=0.6,linewidths=1)
    axi.set_xlim(10.25,11.75)
    axi.set_ylim(-10.25,-8.25)
    stuff=axi.findobj(match=matplotlib.collections.LineCollection)

    #kdeplot adds LineCollections to axis, want to find which ones to show in legend:
    i1=-1*int(len(stuff)/2)-3
    i2=-3
    
    axi.legend([stuff[i1],stuff[i2]],['merger','nonmerger'],fontsize=10)
    
    pyplot.xlabel(r'$log_{10} M_*$')
    pyplot.ylabel(r'$log_{10} SSFR$')



    axi=f7.add_subplot(122)
    axi.locator_params(nbins=5,prune='both')

    ssfr_mergers=rf_dict['logssfr'][m_set_rf]
    mass_mergers=np.log10( rf_dict['Mstar_Msun'][m_set_rf])
    ssfr_nonmergers=rf_dict['logssfr'][nm_set_rf]
    mass_nonmergers=np.log10( rf_dict['Mstar_Msun'][nm_set_rf])

    
    sns.kdeplot(mass_mergers,ssfr_mergers,shade=False,shade_lowest=False,cmap='Blues',alpha=0.6,linewidths=1)
    sns.kdeplot(mass_nonmergers,ssfr_nonmergers,shade=False,shade_lowest=False,cmap='Oranges',alpha=0.6,linewidths=1)
    axi.set_xlim(10.25,11.75)
    axi.set_ylim(-10.25,-8.25)
    stuff=axi.findobj(match=matplotlib.collections.LineCollection)
    i1=-1*int(len(stuff)/2)-3
    i2=-3
    
    axi.legend([stuff[i1],stuff[i2]],[r'$P_{RF}\geq  $'+'{:5.3f}'.format(best_th),r'$P_{RF}< $'+'{:5.3f}'.format(best_th)],fontsize=10)
    

    axi.set_yticklabels([])
    
    pyplot.xlabel(r'$log_{10} M_*$')

    f7.savefig(ssfr_filen,dpi=300)
    pyplot.close(f7)     



    


    
    grid_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_gridplot.pdf'
    f6 = pyplot.figure(figsize=(ncol+0.5,ncol+0.5), dpi=300)
    pyplot.subplots_adjust(left=0.11, right=0.99, bottom=0.10, top=0.99,wspace=0.0,hspace=0.0)

    plot_rf_grid(f6,df,rf_probs,rf_class,rf_flag,cols,plotlabels=plotlabels,rfthresh=best_th)
    
    f6.savefig(grid_filen,dpi=300)
    pyplot.close(f6)     




    
    prob_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_probvmass.pdf'
    f7 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.98,wspace=0.0,hspace=0.0)
    sp=10


    axi4=f7.add_subplot(1,1,1)
    axi4.locator_params(nbins=5,prune='both')


    axi4.semilogy(rf_mass,rf_probs,'o',markersize=2,markeredgecolor='None',markerfacecolor='Gray',zorder=99)
    axi4.semilogy(rf_mass[rf_flag==True],rf_probs[rf_flag==True],'o',markersize=5,markerfacecolor='Black',markeredgecolor='Orange',markeredgewidth=0.4,zorder=99)
    axi4.semilogy(rf_mass[rf_class],rf_probs[rf_class],'*g',markersize=3,markeredgecolor='None',zorder=99)

    axi4.semilogy([10,13],[ROCtests['thresh'][best_i2],ROCtests['thresh'][best_i2]],color=ROCtests['thresh_color_frac_error'],lw=0.5,zorder=1)
    axi4.semilogy([10,13],[ROCtests['thresh'][best_i3],ROCtests['thresh'][best_i3]],color=ROCtests['thresh_color_balance_point'],lw=0.5,zorder=1)
    axi4.semilogy([10,13],[ROCtests['thresh'][best_i4],ROCtests['thresh'][best_i4]],color=ROCtests['thresh_color_mcc'],lw=0.5,zorder=1)
                
    leg=pyplot.legend(['nonmerger','merger',r'$P_{RF}>$'+'{:5.3f}'.format(best_th)],fontsize=sp,
                      markerscale=1,scatterpoints=3,numpoints=3,framealpha=1.0,loc='lower right')
    leg.set_zorder(102)
    
    
    axi4.set_xlim(10.4,12.2)
    axi4.set_ylim(0.015,1.30)
    axi4.set_xlabel(r'log$_{10} M_*/M_{\odot}$')
    axi4.set_ylabel(r'$P_{RF}$')
    axi4.tick_params(labelsize=sp)
    
    
    f7.savefig(prob_filen,dpi=300)
    pyplot.close(f7)     




    #saves the output as a file
    if not os.path.lexists('rfoutput'):
        os.mkdir('rfoutput')
    if not os.path.lexists('rfoutput/'+labelfunc):
        os.mkdir('rfoutput/'+labelfunc)


    prob_df=pandas.DataFrame(rf_probs)
        
    #is this the right df?
    #labels['mergerFlag']=df['mergerFlag']
    prob_df['mergerFlag']=df['mergerFlag']
    #labels['SubfindID']=df['SubfindID']
    prob_df['SubfindID']=df['SubfindID']
    prob_df['BestThresh']=best_th*np.ones_like(rf_probs)
                
    df.to_pickle('rfoutput/'+labelfunc+'/'+rflabel+'_data_{}_{}_{}.pkl'.format(sk,fk,seed))
    
    #test_feature_importances.to_pickle('rfoutput/'+labelfunc+'/'+rflabel+'_importances_{}_{}.pkl'.format(sk,fk))
    #prob_df.to_pickle('rfoutput/'+labelfunc+'/'+rflabel+'_probability_{}_{}.pkl'.format(sk,fk))

    np.save(arr=test_feature_importances,file='rfoutput/'+labelfunc+'/'+rflabel+'_importances_{}_{}_{}.npy'.format(sk,fk,seed) )

    #this thing is kinda borked?
    np.save(arr=prob_df,file='rfoutput/'+labelfunc+'/'+rflabel+'_probability_{}_{}_{}.npy'.format(sk,fk,seed) )
    prob_df.to_pickle('rfoutput/'+labelfunc+'/'+rflabel+'_probability_{}_{}_{}.pkl'.format(sk,fk,seed))
    
    np.save(arr=ROCstats,file='rfoutput/'+labelfunc+'/'+rflabel+'_rocstats_{}_{}_{}.npy'.format(sk,fk,seed) )  #all
    np.save(arr=ROCtests,file='rfoutput/'+labelfunc+'/'+rflabel+'_roctests_{}_{}_{}.npy'.format(sk,fk,seed) )  #testDF
    np.save(arr=ROCtrain,file='rfoutput/'+labelfunc+'/'+rflabel+'_roctrain_{}_{}_{}.npy'.format(sk,fk,seed) )  #trainDF

    #Save RF objects eventually
    #dummy class for saving rf object.. it works?
    
    rfo=simple_forest(rfc)
    np.save(arr=rfo,file='rfoutput/'+labelfunc+'/'+rflabel+'_rfobj_{}_{}_{}.npy'.format(sk,fk,seed))

    
    
    
    return rflabel, cols, plotlabels, best_th



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

    tp = np.where( (labels==True)*(classifications==True) )[0].shape[0]
    fn = np.where( (labels==True)*(classifications==False))[0].shape[0]
    fp = np.where( (labels==False)*(classifications==True))[0].shape[0]

    if tp+fp == 0:
        ppv = 0
    else:
        ppv = 1.*tp/(tp+fp)

    #if tn+fn ==0:
    #    npv=1.0
    #else:
    #    npv = 1.*tn/(tn+fn)
    
    completeness = float(tp)/float(tp + fn)

    Ntp= np.sum(labels)
    Ntn= labels.shape[0]-Ntp

    tpr=(1.0*tp)/float(Ntp)
    fpr=(1.0*fp)/float(Ntn)
    
    return completeness, ppv, tpr, fpr

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




def label_merger_window500_major(merF,sk):
    latest_NumMajorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    label_boolean = latest_NumMajorMergerLast500Myr >= 1.0
    return label_boolean,latest_NumMajorMergerLast500Myr    


#assumes merger_span = 0.25 Gyr so this covers 500Myr around image
def label_merger_window500_both(merF,sk):
    latest_NumMajorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    latest_NumMinorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast500Myr')
    merger_number = latest_NumMajorMergerLast500Myr + latest_NumMinorMergerLast500Myr

    label_boolean = np.logical_or( latest_NumMajorMergerLast500Myr >= 1.0, latest_NumMinorMergerLast500Myr >= 1.0 )
    #print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number

#assumes merger_span = 0.25 Gyr so this covers 250Myr after image
def label_merger_forward250_both(merF,sk):
    latest_NumMajorMergerLast250Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast250Myr')
    latest_NumMinorMergerLast250Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast250Myr')
    merger_number = latest_NumMajorMergerLast250Myr + latest_NumMinorMergerLast250Myr

    label_boolean = np.logical_or( latest_NumMajorMergerLast250Myr >= 1.0, latest_NumMinorMergerLast250Myr >= 1.0 )
    #print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number


def label_merger_forward500_both_new(merF,sk):
    this_NumMajorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast250Myr')
    this_NumMinorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast250Myr')
    
    this_snapNextMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMajorMerger')
    this_snapNextMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMinorMerger')

    this_snapLastMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMajorMerger')
    this_snapLastMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMinorMerger')
        
    latest_NumMajorMergersLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    latest_NumMinorMergersLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast500Myr')

    merger_number=latest_NumMajorMergersLast500Myr + latest_NumMinorMergersLast500Myr
    
    sk_t = gsu.age_at_snap(int(sk[-3:]))

    snaplist=np.arange(0,136,1)
    agelist=np.zeros_like(snaplist*1.0)
    
    for sn in snaplist:
        agelist[sn]=gsu.age_at_snap(sn)

    t_lastMaj=np.where(this_snapLastMajorMerger != -1, agelist[this_snapLastMajorMerger], agelist[this_snapLastMajorMerger]*0.0-100.0)
    t_lastMin=np.where(this_snapLastMinorMerger != -1, agelist[this_snapLastMinorMerger], agelist[this_snapLastMinorMerger]*0.0-100.0)
    t_nextMaj=np.where(this_snapNextMajorMerger != -1, agelist[this_snapNextMajorMerger], agelist[this_snapNextMajorMerger]*0.0+100.0)
    t_nextMin=np.where(this_snapNextMinorMerger != -1, agelist[this_snapNextMinorMerger], agelist[this_snapNextMinorMerger]*0.0+100.0)
    
    label_boolean = np.logical_or(np.logical_or(np.logical_or((t_lastMaj >= sk_t-0.0),
                                                              (t_lastMin >= sk_t-0.0)),
                                                (t_nextMaj <= sk_t+0.5),),
                                  (t_nextMin <= sk_t+0.5 ))

    #for i in np.arange(100):
    #    print(merger_number[i],t_lastMaj[i],t_lastMin[i],t_nextMaj[i],t_nextMin[i],label_boolean[i],sk_t)

    
    return label_boolean,merger_number



def label_merger_window500_both_new(merF,sk):
    this_NumMajorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast250Myr')
    this_NumMinorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast250Myr')
    
    this_snapNextMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMajorMerger')
    this_snapNextMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMinorMerger')

    this_snapLastMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMajorMerger')
    this_snapLastMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMinorMerger')
        
    latest_NumMajorMergersLast500Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast500Myr')
    latest_NumMinorMergersLast500Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast500Myr')

    merger_number=latest_NumMajorMergersLast500Myr + latest_NumMinorMergersLast500Myr  #actually "this"
    
    sk_t = gsu.age_at_snap(int(sk[-3:]))

    snaplist=np.arange(0,136,1)
    agelist=np.zeros_like(snaplist*1.0)
    
    for sn in snaplist:
        agelist[sn]=gsu.age_at_snap(sn)

    t_lastMaj=np.where(this_snapLastMajorMerger != -1, agelist[this_snapLastMajorMerger], agelist[this_snapLastMajorMerger]*0.0-100.0)
    t_lastMin=np.where(this_snapLastMinorMerger != -1, agelist[this_snapLastMinorMerger], agelist[this_snapLastMinorMerger]*0.0-100.0)
    t_nextMaj=np.where(this_snapNextMajorMerger != -1, agelist[this_snapNextMajorMerger], agelist[this_snapNextMajorMerger]*0.0+100.0)
    t_nextMin=np.where(this_snapNextMinorMerger != -1, agelist[this_snapNextMinorMerger], agelist[this_snapNextMinorMerger]*0.0+100.0)
    
    label_boolean = np.logical_or(np.logical_or(np.logical_or((t_lastMaj >= sk_t-0.25),
                                                              (t_lastMin >= sk_t-0.25)),
                                                (t_nextMaj <= sk_t+0.25),),
                                  (t_nextMin <= sk_t+0.25 ))

    #for i in np.arange(100):
    #    print(merger_number[i],t_lastMaj[i],t_lastMin[i],t_nextMaj[i],t_nextMin[i],label_boolean[i],sk_t)

    
    return label_boolean,merger_number,t_lastMaj,t_lastMin,t_nextMaj,t_nextMin


def label_merger_window500_major_new(merF,sk):
    this_NumMajorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast250Myr')
    this_NumMinorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast250Myr')
    
    this_snapNextMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMajorMerger')
    this_snapNextMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMinorMerger')

    this_snapLastMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMajorMerger')
    this_snapLastMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMinorMerger')
        
    latest_NumMajorMergersLast500Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast500Myr')
    latest_NumMinorMergersLast500Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast500Myr')

    merger_number=latest_NumMajorMergersLast500Myr  #actually "this"
    
    sk_t = gsu.age_at_snap(int(sk[-3:]))

    snaplist=np.arange(0,136,1)
    agelist=np.zeros_like(snaplist*1.0)
    
    for sn in snaplist:
        agelist[sn]=gsu.age_at_snap(sn)

    t_lastMaj=np.where(this_snapLastMajorMerger != -1, agelist[this_snapLastMajorMerger], agelist[this_snapLastMajorMerger]*0.0-100.0)
    t_lastMin=np.where(this_snapLastMinorMerger != -1, agelist[this_snapLastMinorMerger], agelist[this_snapLastMinorMerger]*0.0-100.0)
    t_nextMaj=np.where(this_snapNextMajorMerger != -1, agelist[this_snapNextMajorMerger], agelist[this_snapNextMajorMerger]*0.0+100.0)
    t_nextMin=np.where(this_snapNextMinorMerger != -1, agelist[this_snapNextMinorMerger], agelist[this_snapNextMinorMerger]*0.0+100.0)
    
    label_boolean = np.logical_or(np.logical_or(np.logical_or((t_lastMaj >= sk_t-0.25),
                                                              (t_lastMaj >= sk_t-0.25)),
                                                (t_nextMaj <= sk_t+0.25),),
                                  (t_nextMaj <= sk_t+0.25 ))

    #for i in np.arange(100):
    #    print(merger_number[i],t_lastMaj[i],t_lastMin[i],t_nextMaj[i],t_nextMin[i],label_boolean[i],sk_t)

    
    return label_boolean,merger_number,t_lastMaj,t_lastMin,t_nextMaj,t_nextMin



def label_merger_window250_both_new(merF,sk):
    this_NumMajorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast250Myr')
    this_NumMinorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast250Myr')
    
    this_snapNextMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMajorMerger')
    this_snapNextMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMinorMerger')

    this_snapLastMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMajorMerger')
    this_snapLastMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMinorMerger')
        
    latest_NumMajorMergersLast250Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast250Myr')
    latest_NumMinorMergersLast250Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast250Myr')

    merger_number=this_NumMajorMergersLast250Myr + this_NumMinorMergersLast250Myr
    
    sk_t = gsu.age_at_snap(int(sk[-3:]))

    snaplist=np.arange(0,136,1)
    agelist=np.zeros_like(snaplist*1.0)
    
    for sn in snaplist:
        agelist[sn]=gsu.age_at_snap(sn)

    t_lastMaj=np.where(this_snapLastMajorMerger != -1, agelist[this_snapLastMajorMerger], agelist[this_snapLastMajorMerger]*0.0-100.0)
    t_lastMin=np.where(this_snapLastMinorMerger != -1, agelist[this_snapLastMinorMerger], agelist[this_snapLastMinorMerger]*0.0-100.0)
    t_nextMaj=np.where(this_snapNextMajorMerger != -1, agelist[this_snapNextMajorMerger], agelist[this_snapNextMajorMerger]*0.0+100.0)
    t_nextMin=np.where(this_snapNextMinorMerger != -1, agelist[this_snapNextMinorMerger], agelist[this_snapNextMinorMerger]*0.0+100.0)
    
    label_boolean = np.logical_or(np.logical_or(np.logical_or((t_lastMaj >= sk_t-0.125),
                                                              (t_lastMin >= sk_t-0.125)),
                                                (t_nextMaj <= sk_t+0.125),),
                                  (t_nextMin <= sk_t+0.125 ))

    #for i in np.arange(100):
    #    print(merger_number[i],t_lastMaj[i],t_lastMin[i],t_nextMaj[i],t_nextMin[i],label_boolean[i],sk_t)

    
    return label_boolean,merger_number



def label_merger_wide_both(merF,sk):
    this_NumMajorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast250Myr')
    this_NumMinorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast250Myr')
    
    this_snapNextMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMajorMerger')
    this_snapNextMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMinorMerger')

    this_snapLastMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMajorMerger')
    this_snapLastMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMinorMerger')
        
    latest_NumMajorMergersLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    latest_NumMinorMergersLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast500Myr')

    merger_number=latest_NumMajorMergersLast500Myr + latest_NumMinorMergersLast500Myr
    
    sk_t = gsu.age_at_snap(int(sk[-3:]))

    snaplist=np.arange(0,136,1)
    agelist=np.zeros_like(snaplist*1.0)
    
    for sn in snaplist:
        agelist[sn]=gsu.age_at_snap(sn)

    t_lastMaj=np.where(this_snapLastMajorMerger != -1, agelist[this_snapLastMajorMerger], agelist[this_snapLastMajorMerger]*0.0-100.0)
    t_lastMin=np.where(this_snapLastMinorMerger != -1, agelist[this_snapLastMinorMerger], agelist[this_snapLastMinorMerger]*0.0-100.0)
    t_nextMaj=np.where(this_snapNextMajorMerger != -1, agelist[this_snapNextMajorMerger], agelist[this_snapNextMajorMerger]*0.0+100.0)
    t_nextMin=np.where(this_snapNextMinorMerger != -1, agelist[this_snapNextMinorMerger], agelist[this_snapNextMinorMerger]*0.0+100.0)
    
    label_boolean = np.logical_or(np.logical_or(np.logical_or((t_lastMaj >= sk_t-0.25),
                                                              (t_lastMin >= sk_t-0.25)),
                                                (t_nextMaj <= sk_t+0.75),),
                                  (t_nextMin <= sk_t+0.75 ))


    
    return label_boolean,merger_number


def label_merger_wide_major(merF,sk):
    this_NumMajorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast250Myr')
    this_NumMinorMergersLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast250Myr')
    
    this_snapNextMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMajorMerger')
    this_snapNextMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumNextMinorMerger')

    this_snapLastMajorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMajorMerger')
    this_snapLastMinorMerger = get_mergerinfo_val(merF,sk,'this_SnapNumLastMinorMerger')
        
    latest_NumMajorMergersLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    latest_NumMinorMergersLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast500Myr')

    merger_number=latest_NumMajorMergersLast500Myr + latest_NumMinorMergersLast500Myr
    
    sk_t = gsu.age_at_snap(int(sk[-3:]))

    snaplist=np.arange(0,136,1)
    agelist=np.zeros_like(snaplist*1.0)
    
    for sn in snaplist:
        agelist[sn]=gsu.age_at_snap(sn)

    t_lastMaj=np.where(this_snapLastMajorMerger != -1, agelist[this_snapLastMajorMerger], agelist[this_snapLastMajorMerger]*0.0-100.0)
    t_lastMin=np.where(this_snapLastMinorMerger != -1, agelist[this_snapLastMinorMerger], agelist[this_snapLastMinorMerger]*0.0-100.0)
    t_nextMaj=np.where(this_snapNextMajorMerger != -1, agelist[this_snapNextMajorMerger], agelist[this_snapNextMajorMerger]*0.0+100.0)
    t_nextMin=np.where(this_snapNextMinorMerger != -1, agelist[this_snapNextMinorMerger], agelist[this_snapNextMinorMerger]*0.0+100.0)
    
    label_boolean = np.logical_or(np.logical_or(np.logical_or((t_lastMaj >= sk_t-0.25),
                                                              (t_lastMaj >= sk_t-0.25)),
                                                (t_nextMaj <= sk_t+0.75),),
                                  (t_nextMaj <= sk_t+0.75 ))


    
    return label_boolean,merger_number



def label_merger_past250_both(merF,sk):
    this_NumMajorMergerLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast250Myr')
    this_NumMinorMergerLast250Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast250Myr')
    merger_number = this_NumMajorMergerLast250Myr + this_NumMinorMergerLast250Myr

    label_boolean = np.logical_or( this_NumMajorMergerLast250Myr >= 1.0, this_NumMinorMergerLast250Myr >= 1.0 )
    #print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number


def label_merger_past500_both(merF,sk):
    this_NumMajorMergerLast500Myr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLast500Myr')
    this_NumMinorMergerLast500Myr = get_mergerinfo_val(merF,sk,'this_NumMinorMergersLast500Myr')
    merger_number = this_NumMajorMergerLast500Myr + this_NumMinorMergerLast500Myr

    label_boolean = np.logical_or( this_NumMajorMergerLast500Myr >= 1.0, this_NumMinorMergerLast500Myr >= 1.0 )
    #print('N mergers per True: ', np.sum(merger_number[label_boolean])/np.sum(label_boolean))

    return label_boolean,merger_number


#!!!!!
#NOTE USE ONLY IF MERGERDATA FILE HAS SPAN=0.5 Gyr i.e. the March2017 numbers not the May2017 one!!!
#assumes merger_span = 0.5 Gyr so this covers 500Myr after image
def label_merger_forward500_both(merF,sk):
    latest_NumMajorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLast500Myr')
    latest_NumMinorMergerLast500Myr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLast500Myr')
    merger_number = latest_NumMajorMergerLast500Myr + latest_NumMinorMergerLast500Myr

    label_boolean = np.logical_or( latest_NumMajorMergerLast500Myr >= 1.0, latest_NumMinorMergerLast500Myr >= 1.0 )

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

def pc_dict_from_candels(df,lab='_I'):

    parameters = ['C','M20','GINI','ASYM','MPRIME','I','D']
    pcd = {}
    pcd['C'] = df['CON'+lab] #get_all_morph_val(msF,sk,fk,'CC')
    pcd['M20'] = df['M20'+lab]#get_all_morph_val(msF,sk,fk,'M20')
    pcd['GINI'] = df['GINI'+lab]#get_all_morph_val(msF,sk,fk,'GINI')
    pcd['ASYM'] = df['ASYM'+lab]#get_all_morph_val(msF,sk,fk,'ASYM')
    pcd['MPRIME'] = df['MPRIME'+lab]#get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
    pcd['I'] = df['I'+lab]#get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
    pcd['D'] = df['D'+lab]#get_all_morph_val(msF,sk,fk,'MID1_DSTAT')

    npmorph = PyML.machinelearning.dataMatrix(pcd,parameters)
    pc = PyML.machinelearning.pcV(npmorph)


    return pc,pcd


def make_rf_evolution_plots(snap_keys_use,fil_keys_use,dz1,dz2,cols,plotlabels,rflabel='paramsmod',rf_labelfunc='label_merger1',rf_masscut=0.0,twin=0.5,candels_stuff=None,style='fix',seed=0):

    labelfunc=rf_labelfunc

    assert(cols is not None)
    assert(plotlabels is not None)
    
    
    plot_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+style+'_global_stats.pdf'
    if not os.path.lexists('rf_plots'):
        os.mkdir('rf_plots')
    if not os.path.lexists('rf_plots/'+labelfunc):
        os.mkdir('rf_plots/'+labelfunc)

        
    f1 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)


    imp_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+style+'_importance_stats.pdf'
    f2 = pyplot.figure(figsize=(5.5,5.5), dpi=300)
    pyplot.subplots_adjust(left=0.20, right=0.98, bottom=0.08, top=0.98,wspace=0.0,hspace=0.0)



    fm_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+style+'_merger_stats.pdf'
    f5 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.22, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)

    err_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+style+'_error_stats.pdf'
    f6 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.22, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)

    fml_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+style+'_merger_stats_light.pdf'
    f7 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.22, right=0.98, bottom=0.15, top=0.96,wspace=0.0,hspace=0.0)

    

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
    axi2.tick_params(axis='both',which='both',labelsize=labs)
    axi2.locator_params(nbins=5,prune='both')

  

    axi5 = f5.add_subplot(1,1,1)
    axi5.set_xlim(0.2,4.5)
    axi5.set_ylim(5.0e-3,5.0)
    axi5.tick_params(axis='both',which='both',labelsize=labs)
    axi5.locator_params(nbins=7,prune='both')

    axi5.set_xlabel(r'$redshift$',size=labs)
    axi5.set_ylabel(r'$fraction$',size=labs)


    axi6 = f6.add_subplot(1,1,1)
    axi6.set_xlim(0.2,4.5)
    axi6.set_ylim(0.02,1.5)
    axi6.tick_params(axis='both',which='both',labelsize=labs)
    axi6.locator_params(nbins=7,prune='both')

    axi6.set_xlabel(r'$redshift$',size=labs)
    axi6.set_ylabel(r'$fractional\ error$',size=labs)


    axi7 = f7.add_subplot(1,1,1)
    axi7.set_xlim(1.25,9.5)
    axi7.set_ylim(1.0e-2,10.0)
    axi7.tick_params(axis='both',which='both',labelsize=labs)
    axi7.locator_params(nbins=7,prune='both')
    
    axi7.set_xlabel(r'$cosmic\ time$ [Gyr]',size=labs)
    axi7.set_ylabel(r'$galaxy\ merger\ rate$ '+'$[Gyr^{-1}]$',size=labs)

    
    #fakez=np.linspace(0.5,5.0,50)
    #ratez=np.zeros_like(fakez)

    #for i,z in enumerate(fakez):
    #    vrg_rate,vrg_err=VRG_dndt(0.10,1.0,10.0**10.5,z)
    #    ratez[i]=vrg_rate
  
    #axi5.semilogy(fakez,ratez*0.5,linestyle='solid',color='black',linewidth=1.5)

    vrg_z=[]
    vrg_rate=[]

    imp_norm = 1.0/float(len(cols))


    #orig_snap_keys = snap_keys
    #orig_fil_keys = fil_keys

    #load CANDELS data
    df1_all,df2_all,df3_all=merge_field.load_all_candels() #load_all_candels(zrange=[dkz1,dkz2])  #f814, f125, f160


    for sk,fk,dz1u,dz2u in zip(snap_keys_use,fil_keys_use,dz1,dz2):

        redshift=gsu.redshift_from_snapshot(int(sk[-3:]))

        if redshift > 4.2:
            continue
        
        ins=fk.split('-')[0]
        
        rfdata = 'rfoutput/'+labelfunc+'/'
        data_file = rfdata+rflabel+'_data_{}_{}_{}.pkl'.format(sk,fk,seed)
        result_file= rfdata+rflabel+'_importances_{}_{}_{}.npy'.format(sk,fk,seed)
        prob_file = rfdata+rflabel+'_probability_{}_{}_{}.pkl'.format(sk,fk,seed)
        obj_file = rfdata+rflabel+'_rfobj_{}_{}_{}.npy'.format(sk,fk,seed)
        roc_file = rfdata+rflabel+'_roctests_{}_{}_{}.npy'.format(sk,fk,seed)
    
        rf_data = np.load(data_file,encoding='bytes')
    
        rf_prob_df = np.load(prob_file)
        rf_prob=rf_prob_df[0]

        rf_arr = np.load(obj_file)
        rf_container=rf_arr.all()
        rfo=rf_container.rfo
        
        rf_asym = rf_data['asym'].values
        rf_asym_H=rf_data['asym_2'].values
        
        rf_flag = rf_data['mergerFlag'].values
        rf_dgm20 = rf_data['dGM20'].values
        rf_dgm20_H = rf_data['dGM20_2'].values

        rf_cc = rf_data['cc'].values
        rf_mstar = rf_data['Mstar_Msun'].values
        rf_number = rf_data['mergerNumber'].values

        mergers_per_true=np.sum(rf_number)/np.sum(rf_flag)

        asym_classifier = rf_asym > 0.25
        asym_classifier_H = rf_asym_H > 0.25

        asym_com,asym_ppv,asym_tpr,asym_fpr = simple_classifier_stats(asym_classifier,rf_flag)

        dgm_classifier = rf_dgm20 > 0.10
        dgm_classifier_H = rf_dgm20_H > 0.10

        dgm_com,dgm_ppv,dgm_tpr,dgm_fpr = simple_classifier_stats(dgm_classifier,rf_flag)

        print('MERGERS PER TRUE',mergers_per_true)

        
        ROCtests=np.load(roc_file).all()

        
        importances = np.load(result_file)

        ppv = ROCtests['ppv']
        tpr = ROCtests['tpr']
        fpr = ROCtests['fpr']
        threshes=ROCtests['thresh']


        ikey='best_i_balance_point'
        tc=ROCtests['thresh_color_balance_point'] #ROCtests['thresh_color_balance_point']
        
        best_i=ROCtests[ikey]
        thresh=threshes[best_i]

        
        
        rf_class = rf_prob >= thresh #rf_prob > thresh_balance_point




        test_feature_importances=np.asarray(importances)
        cols=np.asarray(cols)
    
        feature_importance = 100.0*(test_feature_importances/test_feature_importances.max())
        #sorted_idx=np.argsort(feature_importance)
        pos = np.flipud( np.arange(feature_importance.shape[0]) + .5 )

        for imp,p in zip(feature_importance,pos):
            print(imp,p)
            axi2.plot(redshift,p,'o',markersize=20.0*(0.01*imp)**2,color='Black')

        axi2.set_yticks(pos)
        axi2.set_yticklabels(plotlabels)
        
        axi2.set_xlabel('redshift')


        

        print('redshift: {:3.1f}   filter: {:15s}   RF sample size: {:8d}   # True mergers: {}   Avg Mstar: {}'.format(redshift,fk,rf_asym.shape[0],np.sum(rf_flag),np.median(rf_mstar)))
        vrg_z.append(redshift)
        vrg_rate_z,vrg_err_z=VRG_dndt(0.10,1.0,np.median(rf_mstar),redshift)

        vrg_rate.append(vrg_rate_z)

        #for ck,sym in zip(cols,syms):
        #    imp = np.median(importances[ck])/imp_norm
        #    axi2.plot(redshift,imp,sym,markersize=6)

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
            
        symsiz=9

        axi.plot(redshift,tpr[best_i],marker='o',markersize=symsiz,markerfacecolor=tc,markeredgecolor=tc,linestyle='None')
        #axi.plot(redshift,1.0-fpr[best_i_balance_point],marker=symbol,markersize=symsiz,markerfacecolor=tc,markeredgecolor='None')
        axi.plot(redshift,ppv[best_i],marker='^',markersize=symsiz,markerfacecolor=tc,markeredgecolor=tc,linestyle='None')            

        axi.plot(redshift,asym_tpr,marker='o',markersize=symsiz/3,markerfacecolor='None',markeredgecolor='Gray',linestyle='None')
        #axi.plot(redshift,1.0-asym_fpr,marker='s',markersize=symsiz/2,markerfacecolor='None',markeredgecolor='Green')
        axi.plot(redshift,asym_ppv,marker='^',markersize=symsiz/3,markerfacecolor='None',markeredgecolor='Gray',linestyle='None')

        axi.plot(redshift,dgm_tpr,marker='o',markersize=symsiz/3,markerfacecolor='None',markeredgecolor='Green',linestyle='None')
        #axi.plot(redshift,1.0-asym_fpr,marker='s',markersize=symsiz/2,markerfacecolor='None',markeredgecolor='Green')
        axi.plot(redshift,dgm_ppv,marker='^',markersize=symsiz/3,markerfacecolor='None',markeredgecolor='Green',linestyle='None')


        axi.legend(['TPR(RF)','PPV(RF)','TPR(A)','PPV(A)','TPR(GMS)','PPV(GMS)'],loc='upper center',ncol=3,fontsize=8,handletextpad=0)



        
        df1,df2,df3=merge_field.select_data(df1_all,df2_all,df3_all,zrange=[dz1u,dz2u],snpix_limit=[0.5,3.0,3.0],mrange=[10.50,15.0])

        candels_df = apply_sim_rf_to_data(sk,fk,cols,df1,df2,df3,best_i,rflabel=rflabel,labelfunc=labelfunc)
        #includes RF_PROB column

        print(candels_df.columns)
        
        sim_asym_fraction=np.sum(asym_classifier)/asym_classifier.shape[0]
        if dgm_tpr > 0.0:
            sim_dgm_fraction=np.sum(dgm_classifier)/dgm_classifier.shape[0]
        else:
            sim_dgm_fraction=0.0



        candels_class=candels_df['RF_PROB'] >= thresh
        candels_f_rf=np.sum(candels_class)/candels_class.shape[0]
        candels_asym=candels_df['ASYM_I'] >= 0.25
        candels_f_a = np.sum(candels_asym)/candels_class.shape[0]
        candels_gms = candels_df['GMS_I'] >= 0.10
        candels_f_s = np.sum(candels_gms)/candels_class.shape[0]
        

        candels_asym_H=candels_df['ASYM_H'] >= 0.25
        candels_f_a_H = np.sum(candels_asym_H)/candels_class.shape[0]
        candels_gms_H = candels_df['GMS_H'] >= 0.10
        candels_f_s_H = np.sum(candels_gms_H)/candels_class.shape[0]
        
        
            
        #axi5.semilogy(redshift,sim_asym_fraction,'o',markersize=symsiz/3.0,color='Gray',markerfacecolor='None')
        #axi5.semilogy(redshift,sim_dgm_fraction,'^',markersize=symsiz/3.0,color='Green',markerfacecolor='None')
        
        #multiply by average N mergers in a "true" classification

        rf_fraction=  np.sum(rf_class)/rf_class.shape[0]

        
        axi5.semilogy(redshift,rf_fraction,'s',markersize=5,color=tc,markerfacecolor='None')
        axi5.semilogy(redshift,mergers_per_true*(ppv[best_i]/tpr[best_i])*np.sum(rf_class)/rf_class.shape[0],'o',color=tc,markersize=5)

        axi5.semilogy(redshift,candels_f_rf,'sk',markersize=10,markerfacecolor='None')
        axi5.semilogy(redshift,candels_f_rf*mergers_per_true*ppv[best_i]/tpr[best_i],'ok',markersize=10)

        axi5.semilogy(redshift,candels_f_a*mergers_per_true*asym_ppv/asym_tpr,'o',color='Gray',markersize=3)
        #axi5.semilogy(redshift,candels_f_s*mergers_per_true*dgm_ppv/dgm_tpr,'^',color='Green',markersize=3)


        #axi7.semilogy(redshift,rf_fraction,'s',markersize=5,color=tc,markerfacecolor='None')
        #axi7.semilogy(redshift,mergers_per_true*(ppv[best_i]/tpr[best_i])*np.sum(rf_class)/rf_class.shape[0],'o',color=tc,markersize=5)

        #axi7.semilogy(redshift,candels_f_rf,'sk',markersize=10,markerfacecolor='None')
        twin=0.5
        axi7.semilogy(illcos.age(redshift).value,(1.0/twin)*candels_f_rf*mergers_per_true*ppv[best_i]/tpr[best_i],'ok',markersize=10)

        axi7.semilogy(illcos.age(redshift).value,(1.0/twin)*candels_f_a*mergers_per_true*asym_ppv/asym_tpr,'o',color='Gray',markersize=3)

        gfs17_times=np.asarray([2.5,3.1,3.95,4.55,8.2])
        gfs17_redshifts=np.zeros_like(gfs17_times)
        
        for iit,gt in enumerate(gfs17_times):
            gfs17_redshifts[iit]=z_at_value(illcos.age, gt*u.Gyr)
        gfs17_rates=np.asarray([0.12,0.19,0.165,0.15,0.16])/(2.4*(1.0 + gfs17_redshifts)**(-2))
        print('RATES', gfs17_rates, gfs17_redshifts)
        
        axi7.semilogy(gfs17_times,gfs17_rates,'^',color='Purple',markersize=7)
        
        
        if sk=='snapshot_103':
            #first_legend = plt.legend(handles=[line1], loc=1)

            # Add the legend manually to the current Axes.
            #ax = plt.gca().add_artist(first_legend)
            firstleg= axi5.legend( [r'$f_{RF}(sim)$',r'$f_{merger}(sim)$',r'$N_{RF}(data)$',r'$f_{merger}(data)$',r'$f_{merger}(A\ only)$'],loc='lower right',fontsize=8,handletextpad=0,ncol=2,columnspacing=1)
            first7leg=axi7.legend( [r'machine learning (GFS+ in prep.)',r'classical image-based',r'evolving pair timescales (GFS+ 2017)'],loc='lower center',fontsize=8,handletextpad=0,ncol=1,columnspacing=1,
                                   title='Data:',markerscale=0.5,labelspacing=0.2)
        
        rf_fractional_error_sq = (1.0/candels_class.shape[0] + 1.0/np.sum(candels_class) + 1.0/np.sum(rf_flag) + 1.0/np.sum(rf_class))
        
        asym_fractional_error_sq=(1.0/candels_asym.shape[0] + 1.0/np.sum(candels_asym) + 1.0/np.sum(rf_flag) + 1.0/np.sum(asym_classifier) )
        asym_fractional_error_sq_H=(1.0/candels_asym_H.shape[0] + 1.0/np.sum(candels_asym_H) + 1.0/np.sum(rf_flag) + 1.0/np.sum(asym_classifier_H) )
        
        if dgm_tpr > 0.0:
            dgm_fractional_error_sq =(1.0/candels_gms.shape[0] + 1.0/np.sum(candels_gms) + 1.0/np.sum(rf_flag) + 1.0/np.sum(dgm_classifier) )
            dgm_fractional_error_sq_H =(1.0/candels_gms_H.shape[0] + 1.0/np.sum(candels_gms_H) + 1.0/np.sum(rf_flag) + 1.0/np.sum(dgm_classifier_H) )

        else:
            dgm_fractional_error_sq =0.0
            dgm_fractional_error_sq_H =0.0

        axi6.semilogy(redshift,rf_fractional_error_sq**0.5,'s',markersize=symsiz,color=tc)
        axi6.semilogy(redshift,asym_fractional_error_sq**0.5,'o',markersize=symsiz/3.0,color='Gray',markerfacecolor='None')
        axi6.semilogy(redshift,dgm_fractional_error_sq**0.5,'^',markersize=symsiz/3.0,color='Green',markerfacecolor='None')

        axi6.legend(['RF classifier (I + H)','Asymmetry (I) alone', 'GMS (I) alone','Asymmetry (H) alone', 'GMS (H) alone'],loc='lower right',fontsize=8)
        
        axi6.semilogy(redshift,asym_fractional_error_sq_H**0.5,'o',markersize=symsiz/9.0,color='Gray',markerfacecolor='None')
        axi6.semilogy(redshift,dgm_fractional_error_sq_H**0.5,'^',markersize=symsiz/9.0,color='Green',markerfacecolor='None')
        
        print(rf_fractional_error_sq,asym_fractional_error_sq,dgm_fractional_error_sq)
        #if sk=='snapshot_103':
        #    axi5.legend([ 'A > 0.25','dGM > 0.1', 'RF model (thresh=0.4)',r'$f_{merge}$'],loc='lower center',fontsize=10,numpoints=1)



        
    f1.savefig(plot_filen,dpi=300)
    pyplot.close(f1)

    f2.savefig(imp_filen,dpi=300)
    pyplot.close(f2)

    #500 Myr window implies predicted merger fraction is R*t_window
    vrg,=axi5.semilogy(np.asarray(vrg_z),np.asarray(vrg_rate)*twin,linestyle='solid',color='black',linewidth=1.5)
    axi5.legend((vrg,),['R-G15'],loc='upper left',fontsize=10,numpoints=1)
    axi5.add_artist(firstleg)

    vrg,=axi7.semilogy(illcos.age(np.asarray(vrg_z)).value,np.asarray(vrg_rate),linestyle='solid',color='black',linewidth=1.5)
    axi7.legend((vrg,),['Theory'],loc='upper left',fontsize=10,numpoints=1)
    axi7.add_artist(first7leg)
    
    f5.savefig(fm_filen,dpi=300)
    pyplot.close(f5)
    
    f6.savefig(err_filen,dpi=300)
    pyplot.close(f6)
    
    f7.savefig(fml_filen,dpi=300)
    pyplot.close(f7)    
    return




def apply_sim_rf_to_data(snap_key,fil_key,cols,df1,df2,df3, best_i, dkz1=1.25,dkz2=1.75,rflabel='paramsmod',rf_masscut=0.0,labelfunc='label_merger1',seed=0):

    sk=snap_key
    fku=fil_key
    
    rfdata = 'rfoutput/'+labelfunc+'/'

    #load RF object/results
    data_file = rfdata+rflabel+'_data_{}_{}_{}.pkl'.format(sk,fku,seed)
    result_file= rfdata+rflabel+'_importances_{}_{}_{}.npy'.format(sk,fku,seed)
    prob_file = rfdata+rflabel+'_probability_{}_{}_{}.npy'.format(sk,fku,seed)
    obj_file = rfdata+rflabel+'_rfobj_{}_{}_{}.npy'.format(sk,fku,seed)
    roc_file = rfdata+rflabel+'_roctests_{}_{}_{}.npy'.format(sk,fku,seed)
    
    rf_data = np.load(data_file,encoding='bytes')
    
    rf_prob_arr = np.load(prob_file)
    rf_prob=rf_prob_arr[:,0]
    rf_flag = rf_prob_arr[:,1]
    
    rf_arr = np.load(obj_file)
    rf_container=rf_arr.all()
    rfo=rf_container.rfo
    print(rfo)

    rf_number = rf_data['mergerNumber']
    

    #APPLY to data

    if rflabel=='twofilters_snp':
        #apply to df1 and df3.. need to x-match on CANDELS_ID
        mergedf=pandas.merge(df1,df3,on='CANDELS_ID')  #x=F814W, y=F160W
        print(mergedf.columns)
        print(cols)

        
        datacols=['GMS_I','GMF_I','ASYM_I','logD_I','CON_I','SN_PIX_I','GMS_H','GMF_H','ASYM_H','logD_H','CON_H','SN_PIX_H']
    elif rflabel=='twofilters':
        mergedf=pandas.merge(df1,df3,on='CANDELS_ID')  #x=F814W, y=F160W
        datacols=['GMS_I','GMF_I','ASYM_I','logD_I','CON_I','GMS_H','GMF_H','ASYM_H','logD_H','CON_H']


    candels_values=mergedf[datacols].values
        
    candels_df=mergedf


    rocdata=np.load(roc_file).all()

    #best_i=rocdata['best_i_balance_point']#tpr=1-fpr
    
    thresh=rocdata['thresh'][best_i]

    ppv = rocdata['ppv'][best_i]
    tpr = rocdata['tpr'][best_i]

    print(candels_df['D_I'])
    print(np.min(candels_df['D_I']))
    
    candels_prob=rfo.predict_proba(candels_values)[:,1]
    candels_df['RF_PROB']=candels_prob
    candels_df['RF_CLASS']=candels_prob >= thresh
    
    rf_total=rf_prob.shape[0]
    rf_class=np.sum(rf_prob >= thresh)
    rf_real =np.sum(rf_flag==True)

    mergerspertrue=np.sum(rf_number[rf_flag==True])/float( np.sum(rf_flag==True))

    
    candels_total=candels_prob.shape[0]
    candels_class=np.sum(candels_prob >= thresh)
    candels_asym=np.sum(candels_df['ASYM_I'] >= 0.25) 
    candels_gms=np.sum(candels_df['GMS_I'] >= 0.10)
    
    print(mergerspertrue, candels_asym, candels_class, candels_total,thresh)
    print('tru merger   fraction: {:5.3f}'.format( float(rf_real)/rf_total ))
    print('RF  rfclass  fraction: {:5.3f}'.format( float(rf_class)/rf_total ))
    print('RF  merger   fraction: {:5.3f}'.format( float(rf_class)*(ppv/tpr)/rf_total ))
    print('HST rfclass  fraction: {:5.3f}'.format( float(candels_class)/candels_total ))
    print('HST merger   fraction: {:5.3f}'.format( float(candels_class)*(ppv/tpr)/candels_total))
    print('HST asym     fraction: {:5.3f}'.format( float(candels_asym)/candels_total ))
    print('HST GMS      fraction: {:5.3f}'.format( float(candels_gms)/candels_total ))


    ncol=len(datacols)
    grid_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fku+'_candelsgrid.pdf'
    f6 = pyplot.figure(figsize=(ncol+0.5,ncol+0.5), dpi=300)
    pyplot.subplots_adjust(left=0.11, right=0.99, bottom=0.10, top=0.99,wspace=0.0,hspace=0.0)

    plot_rf_grid(f6,candels_df,candels_prob,None,None,datacols)
    
    f6.savefig(grid_filen,dpi=300)
    pyplot.close(f6)     


    
        
    #save results in same folder as above *dataprobs* ?
    candels_df.to_pickle(rfdata+rflabel+'_candelsclass_{}_{}_{}.npy'.format(sk,fku,seed))

    
    return candels_df



def plot_rf_grid(f6,rf_data,rfprob,rf_class,rf_flag,cols,plotlabels=None,rfthresh=0.4):

    sp=len(cols)+5

    ncols=len(cols)

    if plotlabels is None:
        plotlabels=cols
    
    cf=0.02
    for i,cn in enumerate(plotlabels):
        axi4=f6.add_subplot(ncols,ncols,i+1)
        axi4.locator_params(nbins=4,prune='both')
        
        axi4.semilogy(rf_data[cols[i]],rfprob,'o',markersize=1,markeredgecolor='None',markerfacecolor='Gray')
        if rf_flag is not None:
            axi4.semilogy(rf_data[cols[i]][rf_flag==True],rfprob[rf_flag==True],'o',markersize=3,markerfacecolor='Black',markeredgecolor='Orange',markeredgewidth=0.4)
        if rf_class is not None:
            axi4.semilogy(rf_data[cols[i]][rf_class],rfprob[rf_class],'*g',markersize=3,markeredgecolor='None')

        xs=sorted(rf_data[cols[i]])
        ys=sorted(rfprob)
    
        axi4.semilogy([xs[int(cf*len(xs))],xs[int((1-cf)*len(xs))]],[rfthresh,rfthresh],marker='None',linestyle='solid',lw=1.0,color='Blue')

        axi4.set_xlim(xs[int(cf*len(xs))],xs[int((1-cf)*len(xs))])
            
        axi4.set_ylim(0.015,1.10)

        if i==0:
            pyplot.legend(['nonmerger','merger',r'$P_{RF}>$'+'{:5.3f}'.format(rfthresh)],fontsize=sp+5,markerscale=4,scatterpoints=3,numpoints=3,framealpha=1.0,bbox_to_anchor=(0.35, 0.50),bbox_transform=pyplot.gcf().transFigure)
            axi4.set_ylabel(r'$P_{RF}$',size=sp)
            axi4.tick_params(labelsize=sp/2)
            axi4.set_xlabel(cn,size=sp)
        else:
            axi4.set_yticklabels([])
            axi4.set_xticklabels([])


            
    n=0
    for i,cni in enumerate(plotlabels):
        for j,cnj in enumerate(plotlabels):
            n=n+1
            if i<=j:
                continue
            
            axi4=f6.add_subplot(ncols,ncols,ncols+1+j*(ncols)+i)
            axi4.locator_params(nbins=4,prune='both')
           
            axi4.plot(rf_data[cols[i]],rf_data[cols[j]],'o',markersize=1,markeredgecolor='None',markerfacecolor='Gray')
            if rf_flag is not None:
                axi4.plot(rf_data[cols[i]][rf_flag==True],rf_data[cols[j]][rf_flag==True],'o',markersize=3,markerfacecolor='Black',markeredgecolor='Orange',markeredgewidth=0.4)
            if rf_class is not None:
                axi4.plot(rf_data[cols[i]][rf_class],rf_data[cols[j]][rf_class],'*g',markersize=3,markeredgecolor='None')
           
            #axi4.set_xlabel(cn,size=sp)
            axi4.tick_params(labelsize=sp/2)
           
            if j==i-1:
                #axi4.set_ylabel(cnj,size=sp)
                axi4.set_xlabel(cni,size=sp)
            else:
                axi4.set_yticklabels([])

            xs=sorted(rf_data[cols[i]])
            ys=sorted(rf_data[cols[j]])
            axi4.set_xlim(xs[int(cf*len(xs))],xs[int((1-cf)*len(xs))])
            axi4.set_ylim(ys[int(cf*len(xs))],ys[int((1-cf)*len(ys))])

    return
    








def make_merger_images(msF,merF,snap_key,fil_key,fil2_key=None,bd='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/',
                       rflabel='paramsmod',rf_masscut=0.0,labelfunc='label_merger1',Npix=None,ckpc=75.0,ckpcz=None,seed=0,**kwargs):

    j=-1
    sk=snap_key
    fku=fil_key
    fk=fku
    
    plotdir = 'images/'+labelfunc
    if fil2_key is None:
        plot_filen = plotdir+'/'+sk+'/'+rflabel+'_mergers_'+sk+'_'+fku
    else:
        plot_filen = plotdir+'/'+sk+'/'+rflabel+'_mergers_'+sk+'_'+fku+'_'+fil2_key
        
    if not os.path.lexists(plotdir):
        os.mkdir(plotdir)
    if not os.path.lexists(plotdir+'/'+sk):
        os.mkdir(plotdir+'/'+sk)
    if not os.path.lexists(plotdir+'/'+sk+'/masshistory'):
        os.mkdir(plotdir+'/'+sk+'/masshistory')        

    N_columns = 12
    N_rows = N_columns

    N_each=int( N_columns/2)
    
    N_pix = Npix #im_npix[j]  #140

    
    f1 = pyplot.figure(figsize=(float(N_columns),float(N_rows)), dpi=600)
    pyplot.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,wspace=0.0,hspace=0.0)
  

    #pick filter to select PC1 values
    b_fk = fku #im_fil_keys[sk]['r'][0]
    g_fk = fku #im_fil_keys[sk]['g'][0]
    if fil2_key is None:
        r_fk = fku #im_fil_keys[sk]['b'][0]
    else:
        r_fk = fku

    #parameters, pcd, pc, pcd = make_pc_dict(msF,sk,r_fk)
    #pc1 = pc.X[:,0].flatten()  #all fk's pc1 values for this snapshot
    mag = get_all_morph_val(msF,sk,r_fk,'MAG')
    mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        
    r_imf = np.asarray(get_all_morph_val(msF,sk,r_fk,'IMFILES'),dtype='str')
    g_imf = np.asarray(get_all_morph_val(msF,sk,g_fk,'IMFILES'),dtype='str')
    b_imf = np.asarray(get_all_morph_val(msF,sk,b_fk,'IMFILES'),dtype='str')

    #select indices  whose images we want
    

    forest_imf = np.asarray(get_all_morph_val(msF,sk,fku,'IMFILES'),dtype='str')
        
    asym = get_all_morph_val(msF,sk,fku,'ASYM')
    gini = get_all_morph_val(msF,sk,fku,'GINI')
    m20 = get_all_morph_val(msF,sk,fku,'M20')
    cc = get_all_morph_val(msF,sk,fku,'CC')
    Mstat = get_all_morph_val(msF,sk,fku,'MID1_MPRIME')
    Istat = get_all_morph_val(msF,sk,fku,'MID1_ISTAT')
    Dstat = get_all_morph_val(msF,sk,fku,'MID1_DSTAT')

    sfid = get_all_snap_val(msF,sk,'SubfindID')
        
    S_GM20 = SGM20(gini,m20)
    F_GM20 = FGM20(gini,m20)


    #print(Istat.shape,mstar.shape,sfid.shape)# looks like I synced the array shapes -- good!
    #gi = np.where(np.isfinite(S_GM20)*np.isfinite(F_GM20)*np.isfinite(asym)*np.isfinite(Mstat)*np.isfinite(Istat)*np.isfinite(Dstat)*np.isfinite(cc)*(mstar >= rf_masscut) != 0)[0]

    
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


        
    redshift = msF['nonparmorphs'][sk][fku]['CAMERA0']['REDSHIFT'].value[0]
    ins = fku.split('-')[0]
    t_sk=illcos.age(redshift).value
    
    if redshift > 4.2:
        return -1

    rfdata = 'rfoutput/'+labelfunc+'/'
        
    data_file = rfdata+rflabel+'_data_{}_{}_{}.pkl'.format(sk,fku,seed)
    result_file= rfdata+rflabel+'_importances_{}_{}_{}.npy'.format(sk,fku,seed)
    prob_file = rfdata+rflabel+'_probability_{}_{}_{}.pkl'.format(sk,fku,seed)
    roc_file = rfdata+rflabel+'_roctests_{}_{}_{}.npy'.format(sk,fku,seed)
    
    rf_data = np.load(data_file,encoding='bytes')
    rf_flag = rf_data['mergerFlag'].values
 
    rf_t_lastMaj = rf_data['t_lastMaj']
    rf_t_lastMin = rf_data['t_lastMin']
    rf_t_nextMaj = rf_data['t_nextMaj']
    rf_t_nextMin = rf_data['t_nextMin']

    probstuff = np.load(prob_file)
        

    
    average_prob=probstuff[0].values
    rf_prob=average_prob
    flags=probstuff['mergerFlag'].values
    rf_sfids=probstuff['SubfindID'].values


    ROCtests=np.load(roc_file).all()

        
    importances = np.load(result_file)

    ppv = ROCtests['ppv']
    tpr = ROCtests['tpr']
    fpr = ROCtests['fpr']
    threshes=ROCtests['thresh']
        
    best_i_balance_point=ROCtests['best_i_balance_point']
    thresh_balance_point=threshes[best_i_balance_point]
    thresh_color_balance_point=ROCtests['thresh_color_balance_point']
    
    rf_class_balance_point = rf_prob > thresh_balance_point
    
    im_file=bd+sk+'/'+rf_data['IMFILE'].values
    if fil2_key is not None:
        im_file=bd+sk+'/'+rf_data['IM2FILE'].values
    
    
    assert(rf_class_balance_point.shape[0]==im_file.shape[0])
    
    #i=0-6,k=0-6:   TP
    #i=0-6,k=7-12:  FP
    #i=7-12,k=0-6:  TN
    #i=7-12,k=7-12:  FN

    tpi= np.logical_and(rf_class_balance_point==True,flags==True)
    fpi= np.logical_and(rf_class_balance_point==True,flags==False)
    tni= np.logical_and(rf_class_balance_point==False,flags==False)
    fni= np.logical_and(rf_class_balance_point==False,flags==True)
    
    tp_dict={'imfiles_r':im_file[tpi],'imfiles_g':im_file[tpi],'imfiles_b':im_file[tpi],
             'rf_prob':rf_prob[tpi],'sfid':rf_sfids[tpi],'rf_flag':flags[tpi],
             'istart':0,'jstart':0,'N':N_each,'sk':sk,'lab':'tp'}
    fp_dict={'imfiles_r':im_file[fpi],'imfiles_g':im_file[fpi],'imfiles_b':im_file[fpi],
             'rf_prob':rf_prob[fpi],'sfid':rf_sfids[fpi],'rf_flag':flags[fpi],
             'istart':0,'jstart':6,'N':N_each,'sk':sk,'lab':'fp'}
    tn_dict={'imfiles_r':im_file[tni],'imfiles_g':im_file[tni],'imfiles_b':im_file[tni],
             'rf_prob':rf_prob[tni],'sfid':rf_sfids[tni],'rf_flag':flags[tni],
             'istart':6,'jstart':0,'N':N_each,'sk':sk,'lab':'tn'}
    fn_dict={'imfiles_r':im_file[fni],'imfiles_g':im_file[fni],'imfiles_b':im_file[fni],
             'rf_prob':rf_prob[fni],'sfid':rf_sfids[fni],'rf_flag':flags[fni],
             'istart':6,'jstart':6,'N':N_each,'sk':sk,'lab':'fn'}


    print('TP   ',sk)
    tp_id,tp_cam=plot_image_classifications(tp_dict,f1,redshift,N_rows,N_columns,Npix=Npix,ckpc=ckpc,ckpcz=ckpcz,plotdir=plotdir,**kwargs)
    print('FP   ',sk)
    fp_id,fp_cam=plot_image_classifications(fp_dict,f1,redshift,N_rows,N_columns,Npix=Npix,ckpc=ckpc,ckpcz=ckpcz,plotdir=plotdir,**kwargs)
    print('TN   ',sk)
    tn_id,tn_cam=plot_image_classifications(tn_dict,f1,redshift,N_rows,N_columns,Npix=Npix,ckpc=ckpc,ckpcz=ckpcz,plotdir=plotdir,**kwargs)
    print('FN   ',sk)
   
    fn_id,fn_cam=plot_image_classifications(fn_dict,f1,redshift,N_rows,N_columns,Npix=Npix,ckpc=ckpc,ckpcz=ckpcz,plotdir=plotdir,**kwargs)


    ax=f1.add_subplot(111)

    ax.set_frame_on(False)
    ax.patch.set_visible(False)
    
    ax.plot([0.0,1.0],[0.5,0.5],linestyle='solid',color=thresh_color_balance_point,lw=5)
    ax.plot([0.5,0.5],[0.0,1.0],linestyle='solid',color=thresh_color_balance_point,lw=5)

    ax.annotate('TP' ,(0.25,0.98),xycoords='axes fraction',ha='center',va='center',color=thresh_color_balance_point,size=30 )
    ax.annotate('FP',(0.75,0.98),xycoords='axes fraction',ha='center',va='center',color=thresh_color_balance_point,size=30 )
    ax.annotate('TN' ,(0.25,0.07),xycoords='axes fraction',ha='center',va='center',color=thresh_color_balance_point,size=30 )
    ax.annotate('FN',(0.75,0.07),xycoords='axes fraction',ha='center',va='center',color=thresh_color_balance_point,size=30 )
                                        
    f1.savefig(plot_filen+'.pdf',dpi=300,facecolor='Black')
    f1.savefig(plot_filen+'.svg',dpi=300,facecolor='Black')

    pyplot.close(f1)


    #PLOT merger timescale distros here?

    plot_filen = plotdir+'/'+rflabel+'_mergertimes_'+sk+'_'+fku+'.pdf'
    f2 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)
    ax=f2.add_subplot(111)



    t_next_merger=np.minimum(rf_t_nextMaj,rf_t_nextMin)
    gti=(t_next_merger < 15.0)

    t_last_merger=np.maximum(rf_t_lastMaj,rf_t_lastMin)
    gti2=(t_last_merger > 0.0)


    t_fp = np.concatenate((t_next_merger[fpi*gti]-t_sk,t_last_merger[fpi*gti2]-t_sk))
    t_fn = np.concatenate((t_next_merger[fni*gti]-t_sk,t_last_merger[fni*gti2]-t_sk))
    print(t_fp)
    print(t_fn)
    
    sns.distplot(t_fp,color='Orange',bins=50)
    sns.distplot(t_fn,color='Black',bins=50)
    
    ax.set_xlim(-1.0,2.0)
    
    f2.savefig(plot_filen,dpi=300)
    pyplot.close(f2)
    
    return sk, tp_id, tp_cam, fp_id, fp_cam, tn_id, tn_cam, fn_id, fn_cam


def plot_image_classifications(info,fig,redshift,N_rows,N_columns,Npix=None,ckpc=75.0,ckpcz=None,alph=0.2,Q=8.0,omd=False,plotdir=None):

    #https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    indf = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    #index function for evenly taking m from n elements

    
    n_things=info['sfid'].shape[0]
    if info['N']**2 > n_things:
        n_take=n_things
    else:
        n_take=info['N']**2

    
    ti=np.flipud( np.argsort(info['rf_prob']))[indf(n_take,n_things)]

    idn=[]
    camn=[]

    sk=info['sk']
    
    k=0
    ki=1
    for this_r,this_g,this_b,this_prob,this_id,this_flag in zip(info['imfiles_r'][ti],info['imfiles_g'][ti],info['imfiles_b'][ti],info['rf_prob'][ti],info['sfid'][ti],info['rf_flag'][ti]):
        k=k+1

        
        r = pyfits.open(this_r)[0].data
        g = pyfits.open(this_g)[0].data
        b = pyfits.open(this_b)[0].data
            
        pixsize_kpc=pyfits.open(this_g)[0].header['PIXKPC']
            
        mid = np.int64(r.shape[0]/2)
        if Npix is not None:
            delt=np.int64(Npix/2)
        elif ckpc is not None:
            this_z=redshift
        
            Npix_use=ckpc/pixsize_kpc
                
            delt=np.int64(Npix_use/2)
        elif ckpcz is not None:
            this_z=redshift
                
            Npix_use=((1.0 + this_z)**(-1.0))*ckpcz/pixsize_kpc
            
            delt=np.int64(Npix_use/2)
        
        r = r[mid-delt:mid+delt,mid-delt:mid+delt]
        g = g[mid-delt:mid+delt,mid-delt:mid+delt]
        b = b[mid-delt:mid+delt,mid-delt:mid+delt]

        r_header = pyfits.open(this_r)[0].header
        this_camnum_int = r_header['CAMERA']


        this_loc= info['jstart'] + (info['istart']+ki-1)*N_columns + (k-1) % info['N']+1
        

        this_cam=this_r.split('_')[-3][-2:]
        idn.append(this_id)
        camn.append(this_cam)
        
        if k==info['N']*ki:
            ki=ki+1

            
        axi = fig.add_subplot(N_rows,N_columns,this_loc)
        axi.set_xticks([]) ; axi.set_yticks([])
            

        rgbthing = make_color_image.make_interactive(b,g,r,alph,Q)
        im=axi.imshow(rgbthing,interpolation='nearest',aspect='auto',origin='lower')
        if omd is True:
            overplot_morph_data(axi,this_g,mid,delt,lw=0.5)
                    
        axi.annotate('${:4.2f}$'.format(this_prob),(0.25,0.10),xycoords='axes fraction',ha='center',va='center',color='White',size=7)
                        
        axi.annotate('${:7d}$'.format(this_id),(0.75,0.15),xycoords='axes fraction',ha='center',va='center',color='White',size=6)

        axi.annotate('${:2d}$'.format(this_camnum_int),(0.75,0.07),xycoords='axes fraction',ha='center',va='center',color='White',size=4 )

        alph=0.2 ; Q=6.0
        relative_path=do_masshistory(sk,this_id,int(this_cam),alph=alph,Q=Q,lab=info['lab'],relpath=plotdir+'/'+sk)
        print(relative_path)
        
        im.set_url(relative_path)
        
        
    return idn, camn


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
        #skip this one in favor of more relevant petrosian-based one?
        #axi.contour(np.transpose(segmap_masked), (clabel-0.0001,), linewidths=lw, colors=('Orange',))


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
        axi.contour(np.transpose(ap_segmap), (10.0-0.0001,), linewidths=lw,colors='DodgerBlue')

    hlist.close()
    

    return axi




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






def plot_rpet_mass(msF,merF,sk,fk,FIG,Cval,gridf='median_grid',rpet=None,mstar=None,nx=4,ny=2,ii=1,i=1,redshift=0.0,skipthis=False,skipann=False,do_x=False,do_y=False,**bin_kwargs):

    if rpet is None:
        rpet  = get_all_morph_val(msF,sk,fk,'RP')
    if mstar is None:
        mstar  = get_all_snap_val(msF,sk,'Mstar_Msun')
    
    bins=18
    labs=10

    xlim=[9.7,11.9]
    ylim=[0.0,1.95]

    axi=FIG.add_subplot(ny,nx,ii)
    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])
    axi.locator_params(nbins=5,prune='both')
        

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
        res,colorobj = gth.make_twod_grid(axi,np.log10(mstar[gi]),rpet[gi],data_dict,gridf,bins=bins,**bin_kwargs)


    anny=0.92
    if skipann is False:
        axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs-1)
        axi.annotate(fk                             ,xy=(0.30,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs-1)


    if do_y is True:
        axi.set_ylabel(r'$log_{10}\ R_P [kpc]$',labelpad=1,size=labs)
    else:
        axi.set_yticklabels([])

    if do_x is True:
        axi.set_xlabel(r'$log_{10}\ M_{*} [M_{\odot}]$',labelpad=1,size=labs)
    else:        
        axi.set_xticklabels([])

        
    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])
    
    return colorobj




def plot_sfr_mass(msF,merF,sk,fk,FIG,Cval,gridf='median_grid',sfr=None,mstar=None,nx=4,ny=2,ii=1,i=1,redshift=0.0,skipthis=False,skipann=False,do_x=False,do_y=False,**bin_kwargs):

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
    axi.locator_params(nbins=5,prune='both')
        
        

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
    if skipann is False:
        axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs-1)
        axi.annotate(fk                             ,xy=(0.30,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs-1)


    if do_y is True:
        axi.set_ylabel(r'$log_{10}\ SFR [M_{\odot}/yr]$',labelpad=1,size=labs)
    else:
        axi.set_yticklabels([])

    if do_x is True:
        axi.set_xlabel(r'$log_{10}\ M_{*} [M_{\odot}]$',labelpad=1,size=labs)
    else:        
        axi.set_xticklabels([])

    return colorobj




def plot_c_a(msF,merF,sk,fk,FIG,Cval,gridf='median_grid',c=None,a=None,nx=4,ny=2,ii=1,i=1,redshift=0.0,skipthis=False,skipann=False,do_x=False,do_y=False,**bin_kwargs):

    if c is None:
        c  = get_all_morph_val(msF,sk,fk,'CC')
    if a is None:
        a  = get_all_morph_val(msF,sk,fk,'ASYM')
    
    bins=18
    labs=10

    xlim=[0.0,0.95]
    ylim=[0.4,5.0]

    axi=FIG.add_subplot(ny,nx,ii)
    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])
    axi.locator_params(nbins=5,prune='both')
        
        

    #if given array, use it
    if type(Cval)==np.ndarray:
        gi = np.where(np.isfinite(Cval)==True)[0]
        data_dict = {'x':Cval[gi]}
    else:
        #otherwise, loop over dict keys and send to plotter
        #this assumes that all values are finite!
        gi = np.arange(a.shape[0])
        data_dict = copy.copy(Cval)

    if skipthis is True:
        pass
        res=None
        colorobj=None
    else:
        res,colorobj = gth.make_twod_grid(axi,a[gi],c[gi],data_dict,gridf,bins=bins,**bin_kwargs)


    anny=0.92
    if skipann is False:
        axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs-1)
        axi.annotate(fk                             ,xy=(0.30,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs-1)


    if do_y is True:
        axi.set_ylabel('$C$',labelpad=1,size=labs)
    else:
        axi.set_yticklabels([])

    if do_x is True:
        axi.set_xlabel('$A$',labelpad=1,size=labs)
    else:        
        axi.set_xticklabels([])
        

    return colorobj





def plot_g_m20(msF,merF,sk,fk,FIG,Cval,gridf='median_grid',g=None,m20=None,nx=4,ny=2,ii=1,i=1,redshift=0.0,skipthis=False,skipann=False,do_x=False,do_y=False,**bin_kwargs):

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
    axi.locator_params(nbins=5,prune='both')

        

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
        res=None
        colorobj=None
    else:
        res,colorobj = gth.make_twod_grid(axi,m20[gi],g[gi],data_dict,gridf,bins=bins,flipx=True,**bin_kwargs)


    anny=0.92

    if skipann is False:
        axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs-1)
        axi.annotate(fk                             ,xy=(0.30,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs-1)


    if do_y is True:
        axi.set_ylabel('$G$',labelpad=1,size=labs)
    else:
        axi.set_yticklabels([])

    if do_x is True:
        axi.set_xlabel('$M_{20}$',labelpad=1,size=labs)
    else:        
        axi.set_xticklabels([])


    return colorobj






def make_structure_plots(msF,merF,snap_keys_use,fil_keys_use,dz1=None,dz2=None,name='pc1',varname=None,bartitle='median PC1',vmin=-2,vmax=3,barticks=[-2,-1,0,1,2,3],
                         rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',rflabel='params',gridf='median_grid',log=False,style='fix',source='sim',seed=0):
    
    if varname is None:
        varname=name

    #start with PC1 or M20
    plot_filen = 'structure_'+source+'/'+labelfunc+'/sfr_mstar_'+name+'_evolution_'+style+'.pdf'
    if not os.path.lexists('structure_'+source):
        os.mkdir('structure_'+source)
    if not os.path.lexists('structure_'+source+'/'+labelfunc):
        os.mkdir('structure_'+source+'/'+labelfunc)
    if not os.path.lexists('structure_'+source+'/'+labelfunc+'/individual'):
        os.mkdir('structure_'+source+'/'+labelfunc+'/individual')    
        
    f1 = pyplot.figure(figsize=(6.5,4.0), dpi=300)
    pyplot.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)


    #G-M20
    plot2_filen = 'structure_'+source+'/'+labelfunc+'/G_M20_'+name+'_evolution_'+style+'.pdf'

    f2 = pyplot.figure(figsize=(6.5,4.0), dpi=300)
    pyplot.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)


    plot3_filen = 'structure_'+source+'/'+labelfunc+'/C_A_'+name+'_evolution_'+style+'.pdf'

    f3 = pyplot.figure(figsize=(6.5,4.0), dpi=300)
    pyplot.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)

    
    plot4_filen = 'structure_'+source+'/'+labelfunc+'/rpet_mstar_'+name+'_evolution_'+style+'.pdf'

    f4 = pyplot.figure(figsize=(6.5,4.0), dpi=300)
    pyplot.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)

    nx=4
    ny=2


    
    i=0
    for sk,fk,dz1u,dz2u in zip(snap_keys_use,fil_keys_use,dz1,dz2):
        i=i+1

        #load morphologies from fk, but use RF catalogs named with rf_keys
        
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

        rpet = get_all_morph_val(msF,sk,fk,'RP')
        pix_arcsec= get_all_morph_val(msF,sk,fk,'PIX_ARCSEC')
        
        sfid = get_all_snap_val(msF,sk,'SubfindID')
        
        S_GM20 = SGM20(gini,m20)
        F_GM20 = FGM20(gini,m20)
        

        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        sfr = get_all_snap_val(msF,sk,'SFR_Msunperyr')

        log_mstar_mhalo = np.log10( mstar/mhalo )

        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
        assert redshift > 0.0
        scale=illcos.kpc_proper_per_arcmin(redshift).value/60.0

        logsize_kpc=np.log10(rpet*pix_arcsec*scale)

        rfdata = 'rfoutput/'+labelfunc+'/'
        
        data_file = rfdata+rflabel+'_data_{}_{}_{}.pkl'.format(sk,fk,seed) #use morphologies from fk
        prob_file = rfdata+rflabel+'_probability_{}_{}_{}.pkl'.format(sk,fk,seed) #use classifications labeled with rfk


        if (varname=='rfprob' or varname=='rf_flag' or varname=='rf_class'):
            if redshift > 4.2:
                skipthis=True
                sfr=None
                mstar=None
                rpet=logsize_kpc
                g=gini
                m20=m20
                cc=cc
                asym=asym
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

                rf_logrpet_kpc=rf_data['size'].values

                probs_df = np.load(prob_file)

                rfprob=probs_df[0].values
                rft=probs_df['BestThresh'].values[0]
                
                rf_class = rfprob >= rft
                print('RFTHRESH: ', rft , np.sum(rf_class), rf_class.shape[0] )

                
                #axi3.plot(roc['fpr'],roc['tpr'])

                sfr=rf_sfr
                mstar=rf_mstar
                rpet=rf_logrpet_kpc
                g=rf_gini
                m20=rf_m20
                cc=rf_cc
                asym=rf_asym
        else:
            skipthis=False
            sfr=None
            mstar=None
            rpet=logsize_kpc
            g=gini
            m20=m20
            cc=cc
            asym=asym


        #defs required: msF,merF,sk,fk,fig,pc1,pc3,rf_class,rf_flag,rf_prob,vmin,vmax,i,nx,ny,sfr,mstar,gridf,rpet,g,m20,cc,asym,ii,redshift,skipthis
        if source=='data':
            df1_all,df2_all,df3_all=merge_field.load_all_candels()
            df1,df2,df3=merge_field.select_data(df1_all,df2_all,df3_all,zrange=[dz1u,dz2u],snpix_limit=[0.5,2.5,2.5],mrange=[9.75,15.0])

            candels_df=df3
            candels_pc,candels_pcd=pc_dict_from_candels(candels_df,lab='_H')
            pc1 = candels_pc.X[:,0].flatten()
            pc3 = candels_pc.X[:,2].flatten()
            PCs=pandas.DataFrame(candels_pc.X)
            mstar=10.0**candels_df['lmass'].values
            sfr=10.0**candels_df['sfr_best'].values
            cc=candels_df['CON_H'].values
            asym=candels_df['ASYM_H'].values
            g=candels_df['GINI_H'].values
            m20=candels_df['M20_H'].values
            rpet=candels_df['RPET_ELL_H'].values
            
        if log is True:
            arr=np.asarray( locals()[varname] )
            arr=np.float64(arr)
            
            Cval = np.log10( arr )
        else:            
            Cval = locals()[varname]



            
        ii=9-i*1



        if i==2*nx or i==nx:
            do_y=True #axi.set_ylabel('$G$',labelpad=1,size=labs)
        else:
            do_y=False
            
        if i<=nx:
            do_x=True
        else:
            do_x=False    #axi.set_xlabel('$M_{20}$',labelpad=1,size=labs)
            

        colorobj=plot_sfr_mass(msF,merF,sk,fk,f1,Cval,vmin=vmin,vmax=vmax,sfr=sfr,mstar=mstar,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis,do_x=do_x,do_y=do_y)

        colorobj=plot_rpet_mass(msF,merF,sk,fk,f4,Cval,vmin=vmin,vmax=vmax,rpet=rpet,mstar=mstar,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis,do_x=do_x,do_y=do_y)
        
        colorobj=plot_g_m20(msF,merF,sk,fk,f2,Cval,vmin=vmin,vmax=vmax,g=g,m20=m20,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis,do_x=do_x,do_y=do_y)

        colorobj=plot_c_a(msF,merF,sk,fk,f3,Cval,vmin=vmin,vmax=vmax,c=cc,a=asym,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis,do_x=do_x,do_y=do_y)

        
        if i==1:
            the_colorobj=copy.copy(colorobj)


        indiv_filen = 'structure_'+source+'/'+labelfunc+'/individual/mass_'+name+'_'+sk+'_'+fk+'_'+style+'.pdf'
        f5 = pyplot.figure(figsize=(3.5,5.5), dpi=300)
        pyplot.subplots_adjust(left=0.14, right=0.98, bottom=0.07, top=0.88,wspace=0.0,hspace=0.0)
        inx=1
        iny=2
        colorobj=plot_sfr_mass(msF,merF,sk,fk,f5,Cval,vmin=vmin,vmax=vmax,sfr=sfr,mstar=mstar,gridf=gridf,ny=iny,nx=inx,ii=1,i=1,redshift=redshift,skipthis=skipthis,do_y=True,do_x=False)
        colorobj=plot_rpet_mass(msF,merF,sk,fk,f5,Cval,vmin=vmin,vmax=vmax,rpet=rpet,mstar=mstar,gridf=gridf,ny=iny,nx=inx,ii=2,i=1,redshift=redshift,skipthis=skipthis,skipann=True,do_y=True,do_x=True)
        gth.make_colorbar(the_colorobj,f5,title=bartitle,ticks=barticks,format='%1d')
        f5.savefig(indiv_filen)
        pyplot.close(f5)

        indiv_filen = 'structure_'+source+'/'+labelfunc+'/individual/morph_'+name+'_'+sk+'_'+fk+'_'+style+'.pdf'
        f5 = pyplot.figure(figsize=(3.5,5.5), dpi=300)
        pyplot.subplots_adjust(left=0.14, right=0.98, bottom=0.07, top=0.88,wspace=0.0,hspace=0.22)
        inx=1
        iny=2
        colorobj=plot_g_m20(msF,merF,sk,fk,f5,Cval,vmin=vmin,vmax=vmax,g=g,m20=m20,gridf=gridf,ny=iny,nx=inx,ii=1,i=1,redshift=redshift,skipthis=skipthis,do_x=True,do_y=True)
        colorobj=plot_c_a(msF,merF,sk,fk,f5,Cval,vmin=vmin,vmax=vmax,c=cc,a=asym,gridf=gridf,ny=iny,nx=inx,ii=2,i=1,redshift=redshift,skipthis=skipthis,do_x=True,do_y=True,skipann=True)
        gth.make_colorbar(the_colorobj,f5,title=bartitle,ticks=barticks,format='%1d')
        f5.savefig(indiv_filen)
        pyplot.close(f5)

        
    
    gth.make_colorbar(the_colorobj,f1,title=bartitle,ticks=barticks,loc=[0.12,0.65,0.17,0.03],format='%1d')

    gth.make_colorbar(the_colorobj,f2,title=bartitle,ticks=barticks,loc=[0.12,0.65,0.17,0.03],format='%1d')

    gth.make_colorbar(the_colorobj,f3,title=bartitle,ticks=barticks,loc=[0.12,0.65,0.17,0.03],format='%1d')

    gth.make_colorbar(the_colorobj,f4,title=bartitle,ticks=barticks,loc=[0.12,0.65,0.17,0.03],format='%1d')


    f1.savefig(plot_filen,dpi=300)
    pyplot.close(f1)

    f2.savefig(plot2_filen,dpi=300)
    pyplot.close(f2)


    f3.savefig(plot3_filen,dpi=300)
    pyplot.close(f3)

    f4.savefig(plot4_filen,dpi=300)
    pyplot.close(f4)

        
    #also show PC3 or Asym

    #if RF results exist, also make that one

    
def make_all_structures(msF,merF,sku,fku,dz1,dz2,rf_labelfunc,rflabel,style,source):

    res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='pc1',bartitle='median PC1',vmin=-2,vmax=3,barticks=[-2,-1,0,1,2,3],labelfunc=rf_labelfunc,rflabel=rflabel,style=style,source=source)
    res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='pc3',bartitle='median PC3',vmin=-1,vmax=2,barticks=[-1,0,1,2],labelfunc=rf_labelfunc,rflabel=rflabel,style=style,source=source)
                
    res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='rfprob',bartitle='$log_{10} <P_{RF}>$',vmin=-2,vmax=0,barticks=[-2,-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=True,rflabel=rflabel,style=style,source=source)
                
    res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='rf_flag',bartitle='$log_{10} f_{merger}$',gridf='log_fraction_grid',vmin=-1,vmax=0,barticks=[-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel,style=style,source=source)
                
    res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='rf_class',bartitle=r'$log_{10} f(P_{RF}\geq th_{\rm best})$',gridf='log_fraction_grid',vmin=-1,vmax=0,barticks=[-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel,style=style,source=source)
                
    #res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='rf_prop',varname='rf_flag',bartitle='merger proportion',gridf='normed_proportion_grid',vmin=0,vmax=1,barticks=[0,1],
    #                           rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel,style=style,source=source)
    res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='rf_prop',varname='rf_flag',bartitle='$log_{10}$ merger proportion',gridf='logged_proportion_grid',vmin=-1,vmax=0,barticks=[-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel,style=style,source=source)                

    res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='pc1',bartitle='median PC1',vmin=-2,vmax=3,barticks=[-2,-1,0,1,2,3],labelfunc=rf_labelfunc,rflabel=rflabel,style=style,source='data')
    res = make_structure_plots(msF,merF,sku,fku,dz1=dz1,dz2=dz2,name='pc3',bartitle='median PC3',vmin=0,vmax=1,barticks=[-1,0,1,2],labelfunc=rf_labelfunc,rflabel=rflabel,style=style,source='data')
                
    
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
            imfki2= np.asarray(im_rf_fil_keys2)==afk

            imfkihst=np.asarray(im_rf_fil_keys_hst)==afk
            imfkihst2=np.asarray(im_rf_fil_keys_hst2)==afk
            
            imski= np.asarray(im_snap_keys)==ask

            #print(imfki,imski,imski[imfki])
            
            if np.sum(imski[imfki2])==1.0:
                spinec='Orange'
                zo=99
                spinelw=3
            elif np.sum(imski[imfki])==1.0:
                spinec='DodgerBlue'
                zo=55
                spinelw=1
            elif np.sum(imski[imfkihst])==1.0:
                spinec='Gray'
                zo=2
                spinelw=1
            elif np.sum(imski[imfkihst2])==1.0:
                spinec='Gray'
                zo=2
                spinelw=3    
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



def do_rf_result_grid(snap_keys_par,fil_keys_par,rflabel='paramsmod',rf_labelfunc='label_merger_forward250_both'):

    i=0

    if rflabel=='paramsmod':
        cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc']
    elif rflabel=='paramsmod2':
        cols=['dGM20','fGM20','asym','Dstat','cc']        
    else:
        assert(False)
        


    fm_filen = 'rf_plots/'+rf_labelfunc+'/'+rflabel+'_mergerdata_stats.pdf'
    f5 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.22, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)


    
    labs=11

    axi5 = f5.add_subplot(1,1,1)
    axi5.set_xlim(0.2,4.5)
    axi5.set_ylim(1.0e-3,2.0)
    axi5.tick_params(axis='both',which='both',labelsize=labs)
    axi5.locator_params(nbins=7,prune='both')

    axi5.set_xlabel(r'$redshift$',size=labs)
    axi5.set_ylabel(r'$fraction$',size=labs)

    alldf_1=pandas.DataFrame()
    alldf_2=pandas.DataFrame()
    alldf_3=pandas.DataFrame()
    
    for sk_rfo,fk_rfo,dkz1,dkz2 in zip(snap_keys_par,fil_keys_par,data_z1_keys,data_z2_keys):

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

        rf_objs=np.load(obj_file) #load RF classifier for snap in top loop


        for sk_app,fk_app in zip(snap_keys_par,fil_keys_par):

            app_snap_int=np.int32(sk_app[-3:])
            app_z = gsu.redshift_from_snapshot(app_snap_int)

            if app_z > 4.2:
                continue
            
            rf_tpr=[]
            rf_fpr=[]

 
            data_file_app = rfdata+rflabel+'_data_{}_{}.pkl'.format(sk_app,fk_app)
            
            df=np.load(data_file_app)
            rf_flag_app = df['mergerFlag']


            
            #apply these classifiers to the application snapshot data frame
            for rfo in rf_objs:
                prob_i = rfo.clrf.predict_proba(df[cols].values)
                pred_i = prob_i[:,1] > 0.3  #??

                 
                rf_tpr.append(np.sum(rf_flag_app[pred_i])/np.sum(rf_flag_app))
                rf_fpr.append( np.sum(np.logical_and(np.asarray(pred_i)==True,np.asarray(rf_flag_app)==False) )/np.sum( np.asarray(rf_flag_app)==False) )
                
            print('RF (z={:4.2f}) applied to simulation at z={:4.2f}.  '.format(rfo_z,app_z), np.mean(np.asarray(rf_tpr)), np.mean(np.asarray(rf_fpr)) )

        df1,df2,df3=load_all_candels(zrange=[dkz1,dkz2])
        if rflabel=='paramsmod':
            datacols=['dGM20','fGM20','ASYM','MPRIME','I','D','CON']
        elif rflabel=='paramsmod2':
            datacols=['dGM20','fGM20','ASYM','D','CON']
            
        rf_data=[]
        if rfo_z <= 1.2:
            df_use=df1  #814
            dfs='1'
        elif rfo_z <=1.7:
            df_use=df2  #125
            dfs='2'
        else:
            df_use=df3  #160
            dfs='3'


        probdata=np.zeros(shape=(df_use.shape[0],5),dtype=np.float32)
        
        for j,rfo in enumerate(rf_objs):
            prob=rfo.clrf.predict_proba(df_use[datacols].values)
            pred = prob[:,1] > 0.3
            rf_data.append(np.sum(pred))
            probdata[:,j]=prob[:,1]

        print(df1.shape,df2.shape,df3.shape,probdata.shape)
        #need to assign these probabilities to the dfs from which they actually derive!@!
        if dfs=='1':
            df1['rfprob']=np.mean(probdata,axis=1)
        elif dfs=='2':
            df2['rfprob']=np.mean(probdata,axis=1)
        else:
            df3['rfprob']=np.mean(probdata,axis=1)
        
        print('RF (z={:4.2f}) applied to data at {:4.2f} < z < {:4.2f}.  Median LogM={:5.2f} ; Median z={:5.2f} '.format(rfo_z,dkz1,dkz2,np.median(df_use['LMSTAR_BC03']),np.median(df_use['Z_BEST'])), np.mean(rf_data), df_use.shape[0])

        frac=np.mean(rf_data)/df_use.shape[0]
        axi5.semilogy(rfo_z,frac,marker='o',color='Blue',markersize=8)

        alldf_1=alldf_1.append(df1)
        alldf_2=alldf_2.append(df2)
        alldf_3=alldf_3.append(df3)
        

    alldf_1.to_pickle(rfdata+rflabel+'_mergerprob_814.df')
    alldf_2.to_pickle(rfdata+rflabel+'_mergerprob_125.df')
    alldf_3.to_pickle(rfdata+rflabel+'_mergerprob_160.df')
    
    f5.savefig(fm_filen,dpi=300)
    pyplot.close(f5)
        

    return




def print_tables(msF,merF,fil_keys,rf_labelfuncs=['label_merger_window500_both','label_merger_forward250_both'],rflabel='paramsmod2',rf_masscut=10.0**(10.5)):

    print(r'\begin{table*}')
    print(r'\centering')
    print(r'\caption{Dataset Properties}')
    print(r'\label{tab:dataset}')
    print(r'\begin{tabular}{ccccccccc}')
    print(r'Snapshot & z & $N_{\rm all}$ & $N_{\rm massive}$ & $N_{\rm images}$ &  $N_{\rm 10:1}$  &  $N_{\rm 4:1}$ & $\left <M/N\right >_{\rm 10:1}$ & $\left <M/N\right >_{\rm 4:1}$ \\')
    print(r' & & $ M_* > 10^{10} M_{\odot}$ & \multicolumn{6}{c}{$ M_* > 10^{10.5} M_{\odot}$}  \\')

    snapshots=['103','085','075','068','064','060','054','049','045','041','038','035']
    for sn in snapshots:
        sk='snapshot_'+sn

        mstar_cam0=get_all_snap_val(msF,sk,'Mstar_Msun',camera='CAMERA0')
        mstar_all=get_all_snap_val(msF,sk,'Mstar_Msun')
        
        gi= mstar_cam0 >= rf_masscut
        gia= mstar_all >= rf_masscut
        
        redshift = msF['nonparmorphs'][sk]['NC-F200W']['CAMERA0']['REDSHIFT'].value[0]

        boolean_flag1,number1,x,x,x,x = eval(rf_labelfuncs[0]+'(merF,sk)')
        boolean_flag2,number2,x,x,x,x = eval(rf_labelfuncs[1]+'(merF,sk)')
        #print(np.sum(boolean_flag1))
        if np.sum(boolean_flag2[gia])==0:
            mpt_1='--'
            mpt_2='--'
        else:
            mpt_1 = str( round( np.sum(number1[gia])/np.sum(boolean_flag1[gia]),1))
            mpt_2 = str( round( np.sum(number2[gia])/np.sum(boolean_flag2[gia]),1))
        
        
        print(r'{:5s} & {:5.1f} & {:8d} & {:8d} & {:8d} & {:8d} & {:8d} & {:5s} & {:5s} \\'.format(sn,redshift,mstar_cam0.shape[0],mstar_cam0[gi].shape[0],mstar_all[gia].shape[0],
                                                                                   np.sum(boolean_flag1[gia]),np.sum(boolean_flag2[gia]),mpt_1,mpt_2))

    print(r'\end{tabular}')
    print(r'\end{table*}')



def do_snapshot_evolution(msF,merF,
                          sku =['snapshot_103','snapshot_085','snapshot_075','snapshot_068','snapshot_064','snapshot_060','snapshot_054','snapshot_049'],
                          fku =['ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ],
                          f2ku=['WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ],
                          dz1 =[0.1, 0.75,1.25,1.75,2.25,2.75,3.5,4.5],
                          dz2 =[0.75,1.25,1.75,2.25,2.75,3.5 ,4.5,5.9],
                          mln =[25,25,25,25,25,25,25,25],
                          labelfunc='label_merger_window500_both',skipcalc=False,rfthresh=0.2,skipstruct=False,skipdata=False,style='fix',skipevo=False,seed=0):


    threshes=[]
    threshes2=[]

    candels_rf_frac=[]
    candels_a_frac =[]
    candels_s_frac =[]
    
    for sk,fk,fk2,dz1u,dz2u,mlnu in zip(sku,fku,f2ku,dz1,dz2,mln):


        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
        if redshift > 4.2:
            threshes.append(0.0)
            continue
        
        rflabel,cols,plotlabels,best_th=simple_random_forest(msF,merF,sk,fk, paramsetting='onefilter',max_leaf_nodes=mlnu,
                                                             rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,style=style,seed=seed)
        rflabel,cols,plotlabels,best_th=simple_random_forest(msF,merF,sk,fk2, paramsetting='onefilter',max_leaf_nodes=mlnu,
                                                             rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,style=style,seed=seed)
        
        rflabel,cols,plotlabels,best_th=simple_random_forest(msF,merF,sk,fk,fil2_key=fk2, paramsetting='twofilters',max_leaf_nodes=mlnu,
                                                             rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,style=style,seed=seed)


        rflabel,cols,plotlabels,best_th=simple_random_forest(msF,merF,sk,fk, paramsetting='onefilter_snp',max_leaf_nodes=mlnu,
                                                             rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,style=style,seed=seed)
        rflabel,cols,plotlabels,best_th=simple_random_forest(msF,merF,sk,fk2, paramsetting='onefilter_snp',max_leaf_nodes=mlnu,
                                                             rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,style=style,seed=seed)

        threshes2.append(best_th)

        rflabel,cols,plotlabels,best_th=simple_random_forest(msF,merF,sk,fk,fil2_key=fk2, paramsetting='twofilters_snp_mi',max_leaf_nodes=mlnu,
                                                             rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,seed=seed)
        
        rflabel,cols,plotlabels,best_th=simple_random_forest(msF,merF,sk,fk,fil2_key=fk2, paramsetting='twofilters_snp',max_leaf_nodes=mlnu,
                                                             rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,seed=seed)

        threshes.append(best_th)

        if skipdata is False:

            sk,tp_id,tp_cam,fp_id,fp_cam,tn_id,tn_cam,fn_id,fn_cam = make_merger_images(msF,merF,sk,fk,rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc=labelfunc,omd=True)
            make_merger_images(msF,merF,sk,fk,fil2_key=fk2,rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc=labelfunc,omd=True)

            plot_mass_history_all(sk,tp_id,tp_cam,fp_id,fp_cam,tn_id,tn_cam,fn_id,fn_cam)

        
    if skipstruct is False:
        res = make_all_structures(msF,merF,sku,fku,dz1,dz2,rf_labelfunc=labelfunc,rflabel=rflabel,style=style+'1',source='sim')
        res = make_all_structures(msF,merF,sku,f2ku,dz1,dz2,rf_labelfunc=labelfunc,rflabel='onefilter_snp',style=style+'2',source='sim')


    if skipevo is False:
        res = make_rf_evolution_plots(sku,fku,dz1,dz2,cols,plotlabels,rf_labelfunc=labelfunc,rflabel=rflabel,style=style)


    
        
    return






def do_filter_images():

    morph_stat_file = os.path.expandvars('$HOME/Dropbox/Workspace/Papers_towrite/Illustris_ObservingMergers/CATALOGS/nonparmorphs_SB25_12filters_all_FILES.hdf5')

    merger_file = os.path.expandvars('$HOME/Dropbox/Workspace/Papers_towrite/Illustris_ObservingMergers/CATALOGS/imagedata_mergerinfo_SB25_2017May08.hdf5')
    merger_file_500 = os.path.expandvars('$HOME/Dropbox/Workspace/Papers_towrite/Illustris_ObservingMergers/CATALOGS/imagedata_mergerinfo_SB25_2017March3.hdf5')


    with h5py.File(morph_stat_file,'r') as msF:
        with h5py.File(merger_file,'r') as merF:
            with h5py.File(merger_file_500,'r') as merF500:
    
                res=make_filter_images(msF,merF,sfid=202,cams='00',snapkey='snapshot_035',alph=0.5,Q=5.0)

    
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

                fil_keys_use=fil_keys_hst2   #  fil_keys #
                
                rflabel='twofilters'
                skipmi=True

                
                print_tables(msF,merF,fil_keys_use,rf_labelfuncs=['label_merger_window500_both_new','label_merger_window500_major_new'],rflabel=rflabel)
                #exit()




                #something like this:
                '''
                do_snapshot_evolution(msF,merF,
                                      sku =['snapshot_075'],
                                      fku =['ACS-F814W'],
                                      f2ku=['WFC3-F160W'],
                                      mln=[25],
                                      labelfunc='label_merger_window500_major_new',skipcalc=True,skipstruct=True,skipdata=False)
                '''
                '''
                do_snapshot_evolution(msF,merF,
                                      sku =['snapshot_075'],
                                      fku =['ACS-F814W'],
                                      f2ku=['WFC3-F160W'],
                                      mln=[25],
                                      labelfunc='label_merger_window500_both_new',skipcalc=False,skipstruct=True,skipdata=True,skipevo=True)
                '''

                labelfunc='label_merger_window500_both_new'
                
                do_snapshot_evolution(msF,merF,
                                      sku =['snapshot_103','snapshot_085','snapshot_075','snapshot_068','snapshot_064','snapshot_060','snapshot_054','snapshot_049'],
                                      fku =['ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ],
                                      f2ku=['WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ],
                                      dz1 =[0.1, 0.75,1.25,1.75,2.25,2.75,3.5,4.5],
                                      dz2 =[0.75,1.25,1.75,2.25,2.75,3.5 ,4.5,5.9],
                                      mln =[50,50,25,25,20,20,10,10],
                                      labelfunc=labelfunc,skipcalc=True,skipdata=False,skipstruct=True,style='fix')

                do_snapshot_evolution(msF,merF,
                                      sku =['snapshot_103','snapshot_085','snapshot_075','snapshot_068','snapshot_064','snapshot_060','snapshot_054','snapshot_049'],
                                      fku =['ACS-F435W'   ,'ACS-F606W'   ,'ACS-F606W'   ,'NC-F115W'   ,'NC-F115W'    ,'NC-F115W'    ,'NC-F115W'    ,'NC-F200W'    ],
                                      f2ku=['ACS-F606W'   ,'ACS-F814W'   ,'NC-F115W'    ,'NC-F150W'   ,'NC-F150W'    ,'NC-F200W'    ,'NC-F200W'    ,'NC-F277W'    ],
                                      dz1 =[0.1, 0.75,1.25,1.75,2.25,2.75,3.5,4.5],
                                      dz2 =[0.75,1.25,1.75,2.25,2.75,3.5 ,4.5,5.9],
                                      mln =[50,50,25,25,20,20,10,10],
                                      labelfunc=labelfunc,skipcalc=True,skipdata=True,skipstruct=True,style='evo')
                
                



                
                
                #res = make_morphology_evolution_plots(msF,merF)
                

                

                #res=do_rf_result_grid(copy.copy(snap_keys),copy.copy(fil_keys_use),rflabel=rflabel,rf_labelfunc='label_merger_window500_both')




                
