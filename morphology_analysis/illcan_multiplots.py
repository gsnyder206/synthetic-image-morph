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

im_rf_fil_keys_hst= ['ACS-F814W','ACS-F814W','NC-F115W','NC-F115W',
                  'WFC3-F160W','WFC3-F160W','WFC3-F160W','WFC3-F160W']

im_rf_fil_keys_hst2= ['ACS-F606W','ACS-F814W','WFC3-F105W','WFC3-F160W',
                  'WFC3-F160W','WFC3-F160W','WFC3-F160W','WFC3-F160W']

im_rf_fil_keys_hst3= ['WFC3-F160W','WFC3-F160W','WFC3-F160W','WFC3-F160W',
                  'WFC3-F160W','WFC3-F160W','WFC3-F160W','WFC3-F160W']


snap_keys = im_snap_keys
fil_keys = im_rf_fil_keys
fil_keys_hst=im_rf_fil_keys_hst
fil_keys_hst2=im_rf_fil_keys_hst2
fil_keys_hst3=im_rf_fil_keys_hst3


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
    


def simple_random_forest(msF,merF,snap_key,fil_key,fil2_key=None,paramsetting='medium',rfiter=3,rf_masscut=10.0**(10.5),labelfunc='label_merger1', thresh=0.4,
                         n_estimators=2000,max_leaf_nodes=-1,max_features=4,balancetrain=True,skip_mi=False,trainfrac=0.67, traininglabel='mergerFlag',skipcalc=False,**kwargs):

    sk=snap_key
    fk=fil_key



    
    
    asym = get_all_morph_val(msF,sk,fk,'ASYM')
    gini = get_all_morph_val(msF,sk,fk,'GINI')
    m20 = get_all_morph_val(msF,sk,fk,'M20')
    cc = get_all_morph_val(msF,sk,fk,'CC')
    Mstat = get_all_morph_val(msF,sk,fk,'MID1_MPRIME')
    Istat = get_all_morph_val(msF,sk,fk,'MID1_ISTAT')
    Dstat = get_all_morph_val(msF,sk,fk,'MID1_DSTAT')
    size = get_all_morph_val(msF,sk,fk,'RP')

    snpix= get_all_morph_val(msF,sk,fk,'SNPIX')
    if fil2_key is not None:
        fk2=fil2_key
        asym_2 = get_all_morph_val(msF,sk,fk2,'ASYM')
        gini_2 = get_all_morph_val(msF,sk,fk2,'GINI')
        m20_2 = get_all_morph_val(msF,sk,fk2,'M20')
        cc_2 = get_all_morph_val(msF,sk,fk2,'CC')
        Mstat_2 = get_all_morph_val(msF,sk,fk2,'MID1_MPRIME')
        Istat_2 = get_all_morph_val(msF,sk,fk2,'MID1_ISTAT')
        Dstat_2 = get_all_morph_val(msF,sk,fk2,'MID1_DSTAT')
        size_2 = get_all_morph_val(msF,sk,fk2,'RP')
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
        cols=['dGM20','fGM20','asym','Dstat','cc','logssfr','size','log_mstar_mhalo','bhmdot','elong','flag','m_a','m_i2','rp','mid1_gini','mid1_m20']        
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
        
    boolean_flag,number = eval(labelfunc+'(merF,sk)')

    mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
    mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
    log_mstar_mhalo = np.log10( mstar/mhalo )
    sfr = get_all_snap_val(msF,sk,'SFR_Msunperyr')
    bhmdot = get_all_snap_val(msF,sk,'BHMdot_Msunperyr')
    
    redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]


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
        rf_dict['Mstat']=Mstat[gi]
        rf_dict['Istat']=Istat[gi]
        rf_dict['size']=np.log10(size[gi])
        rf_dict['Dstat']=np.log10(Dstat[gi])   #NOTE log quantity in paramsmod2
        rf_dict['cc']=cc[gi]

        S_GM20_2 = SGM20(gini_2[gi],m20_2[gi])
        F_GM20_2 = FGM20(gini_2[gi],m20_2[gi])
        
        rf_dict['dGM20_2']=S_GM20_2
        rf_dict['fGM20_2']=F_GM20_2
        rf_dict['asym_2']=asym_2[gi]
        rf_dict['Mstat_2']=Mstat_2[gi]
        rf_dict['Istat_2']=Istat_2[gi]
        rf_dict['size_2']=np.log10(size_2[gi])
        rf_dict['Dstat_2']=np.log10(Dstat_2[gi])   #NOTE log quantity in paramsmod2
        rf_dict['cc_2']=cc_2[gi]
        rf_dict['snp_2']=snpix_2[gi]

        rf_dict['IM2FILE']=f2k_imfile[gi]
        
    else:
        gi = np.where(np.isfinite(gini)*np.isfinite(m20)*np.isfinite(asym)*np.isfinite(Dstat)*np.isfinite(cc)*(mstar >= rf_masscut) != 0)[0]

        S_GM20 = SGM20(gini[gi],m20[gi])
        F_GM20 = FGM20(gini[gi],m20[gi])
        
        rf_dict['dGM20']=S_GM20
        rf_dict['fGM20']=F_GM20
        rf_dict['asym']=asym[gi]
        rf_dict['Mstat']=Mstat[gi]
        rf_dict['Istat']=Istat[gi]
        rf_dict['size']=np.log10(size[gi])
        rf_dict['Dstat']=np.log10(Dstat[gi])   #NOTE log quantity in paramsmod2
        rf_dict['cc']=cc[gi]



    rf_dict['IMFILE']=fk_imfile[gi]
    
    rf_dict['elong']= get_all_morph_val(msF,sk,fk,'ELONG')[gi]
    rf_dict['flag']= get_all_morph_val(msF,sk,fk,'FLAG')[gi]
    rf_dict['m_a']= get_all_morph_val(msF,sk,fk,'M_A')[gi]
    rf_dict['m_i2']= get_all_morph_val(msF,sk,fk,'M_I2')[gi]
    rf_dict['rp']= get_all_morph_val(msF,sk,fk,'RP')[gi]
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



    if max_leaf_nodes==-1:
        max_leaf_nodes= np.int64( 0.5*np.sum(rf_dict['mergerFlag']) )

    
    df=pandas.DataFrame(rf_dict)

    print(rf_dict.keys())
    print(df.columns)
    print(df.describe())
    
    
    trainDF, testDF = PyML.trainingSet(df,training_fraction=trainfrac,seed=0)

    #create cross-validation test
    train = trainDF[cols].values
    test = testDF[cols].values
    labels = trainDF[traininglabel]


    if skipcalc==True:        
        return rflabel,cols
    
    rfc = RandomForestClassifier(n_jobs=-1,oob_score=True,n_estimators=n_estimators,max_leaf_nodes=max_leaf_nodes,
                                 max_features=max_features)  #,class_weight='balanced_subsample'


    rfc.fit(train,labels) 
            
    test_preds = np.asarray(rfc.predict(test))
    test_probs = np.asarray(rfc.predict_proba(test))
    test_feature_importances = rfc.feature_importances_

    print(test_feature_importances)

    
    all_preds = np.asarray(rfc.predict(df[cols].values))
    all_probs = np.asarray(rfc.predict_proba(df[cols].values))

    train_preds = np.asarray(rfc.predict(trainDF[cols].values))
    train_probs = np.asarray(rfc.predict_proba(trainDF[cols].values))
    
    Nroc=100
    threshes= np.logspace(-3.0,0,Nroc)
    ROCstats = {'thresh':np.zeros(Nroc), 'tpr':np.zeros(Nroc),'fpr':np.zeros(Nroc)}
    ROCtests = {'thresh':np.zeros(Nroc), 'tpr':np.zeros(Nroc),'fpr':np.zeros(Nroc)}
    ROCtrain = {'thresh':np.zeros(Nroc), 'tpr':np.zeros(Nroc),'fpr':np.zeros(Nroc)}

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
    for i,at in enumerate(athreshes):
        asym_classifier = testDF['asym'].values > at
        asym_com,asym_ppv,asym_tpr,asym_fpr = simple_classifier_stats(asym_classifier,testDF[traininglabel].values)
        atpr[i]=asym_tpr
        afpr[i]=asym_fpr

    dthreshes=np.linspace(-0.3,0.7,Nroc)
    dtpr=np.zeros_like(dthreshes)
    dfpr=np.zeros_like(dthreshes)
    for i,dt in enumerate(dthreshes):
        dg_classifier = testDF['dGM20'].values > dt
        dg_com,dg_ppv,dg_tpr,dg_fpr = simple_classifier_stats(dg_classifier,testDF[traininglabel].values)
        dtpr[i]=dg_tpr
        dfpr[i]=dg_fpr

        
    roc_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_roc_stats.pdf'
    f3 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)
    axi3=f3.add_subplot(1,1,1)
        
    axi3.plot(ROCstats['fpr'],ROCstats['tpr'],lw=1.0,linestyle='dashed',color='Blue')
    axi3.plot(ROCtests['fpr'],ROCtests['tpr'],lw=2, linestyle='solid',color='Black')
    axi3.plot(ROCtrain['fpr'],ROCtrain['tpr'],lw=1.0,linestyle='dotted',color='Red')

    axi3.plot(afpr,atpr,lw=0.5,linestyle='dotted',color='Gray')
    axi3.plot(dfpr,dtpr,lw=0.5,linestyle='dotted',color='Green')
    
    axi3.legend(['all','cross-val','train','Asymmetry',r'$GMS$'],loc='lower right',fontsize=10)
    
    axi3.set_xlim(-0.05,1.05)
    axi3.set_ylim(-0.05,1.05)
    axi3.set_xlabel('False Positive Rate')
    axi3.set_ylabel('True Positive Rate')
    f3.savefig(roc_filen,dpi=300)
    pyplot.close(f3)



        
    roc_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_roc_thresh.pdf'
    f3 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)
    axi3=f3.add_subplot(1,1,1)
        
    axi3.semilogx(ROCstats['thresh'],ROCstats['tpr'],lw=2,linestyle='solid',color='Blue')
    axi3.semilogx(ROCstats['thresh'],ROCstats['fpr'],lw=2,linestyle='dashed',color='Blue')


    at_normed=(athreshes-athreshes.min(0))/athreshes.ptp(0)
    dt_normed=(dthreshes-dthreshes.min(0))/dthreshes.ptp(0)
    
    axi3.semilogx(at_normed,atpr,lw=0.25,linestyle='dotted',color='Gray')
    axi3.semilogx(at_normed,afpr,lw=0.25,linestyle='dashdot',color='Gray')
    axi3.semilogx(dt_normed,dtpr,lw=0.25,linestyle='dotted',color='Green')
    axi3.semilogx(dt_normed,dfpr,lw=0.25,linestyle='dashdot',color='Green')
    
    axi3.legend(['TPR','FPR','TPR(Asym)','FPR(Asym)','TPR(GMS)','FPR(GMS)'],loc='lower left',fontsize=10)
    
    axi3.set_ylim(-0.05,1.05)
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
    
    rf_class = rf_probs >= thresh
    rf_flag = df[traininglabel].values

    

    ssfr_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_ssfr.pdf'
    f7 = pyplot.figure(figsize=(5.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.17, right=0.95, bottom=0.17, top=0.95,wspace=0.0,hspace=0.0)

    
    axi=f7.add_subplot(121)
    axi.locator_params(nbins=5,prune='both')

    m_set  = rf_flag == True
    nm_set = rf_flag == False

    m_set_rf = rf_probs >= thresh
    nm_set_rf = rf_probs < thresh
    
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
    
    axi.legend([stuff[i1],stuff[i2]],[r'$P_{RF}\geq  $'+'{:4.2f}'.format(thresh),r'$P_{RF}< $'+'{:4.2f}'.format(thresh)],fontsize=10)
    

    axi.set_yticklabels([])
    
    pyplot.xlabel(r'$log_{10} M_*$')

    f7.savefig(ssfr_filen,dpi=300)
    pyplot.close(f7)     



    


    print(df.shape,rf_probs.shape,rf_class.shape,rf_flag.shape,rf_flag[rf_flag==True].shape)
    
    grid_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_gridplot.pdf'
    f6 = pyplot.figure(figsize=(ncol+0.5,ncol+0.5), dpi=300)
    pyplot.subplots_adjust(left=0.11, right=0.99, bottom=0.10, top=0.99,wspace=0.0,hspace=0.0)

    plot_rf_grid(f6,df,rf_probs,rf_class,rf_flag,cols,plotlabels=plotlabels,rfthresh=thresh)
    
    f6.savefig(grid_filen,dpi=300)
    pyplot.close(f6)     





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

                
    df.to_pickle('rfoutput/'+labelfunc+'/'+rflabel+'_data_{}_{}.pkl'.format(sk,fk))
    
    #test_feature_importances.to_pickle('rfoutput/'+labelfunc+'/'+rflabel+'_importances_{}_{}.pkl'.format(sk,fk))
    #prob_df.to_pickle('rfoutput/'+labelfunc+'/'+rflabel+'_probability_{}_{}.pkl'.format(sk,fk))

    np.save(arr=test_feature_importances,file='rfoutput/'+labelfunc+'/'+rflabel+'_importances_{}_{}.npy'.format(sk,fk) )

    np.save(arr=prob_df,file='rfoutput/'+labelfunc+'/'+rflabel+'_probability_{}_{}.npy'.format(sk,fk) )
    
    np.save(arr=ROCstats,file='rfoutput/'+labelfunc+'/'+rflabel+'_rocstats_{}_{}.npy'.format(sk,fk) )  #all
    np.save(arr=ROCtests,file='rfoutput/'+labelfunc+'/'+rflabel+'_roctests_{}_{}.npy'.format(sk,fk) )  #testDF
    np.save(arr=ROCtrain,file='rfoutput/'+labelfunc+'/'+rflabel+'_roctrain_{}_{}.npy'.format(sk,fk) )  #trainDF

    #Save RF objects eventually
    #dummy class for saving rf object.. it works?
    
    rfo=simple_forest(rfc)
    np.save(arr=rfo,file='rfoutput/'+labelfunc+'/'+rflabel+'_rfobj_{}_{}.npy'.format(sk,fk))

    
    
    
    return rflabel, cols





def run_random_forest(msF,merF,snap_keys_par,fil_keys_par,rfiter=3,RUN_RF=True,rf_masscut=10.0**(10.5),labelfunc='label_merger1',balancetrain=True,skip_mi=False):

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
            
            rf_dict['Dstat']=np.log10(Dstat[gi])   #NOTE log quantity in paramsmod2

            rf_dict['cc']=cc[gi]
            rf_dict['mergerFlag']=boolean_flag[gi]
            rf_dict['SubfindID']=sfid[gi]
            rf_dict['Mstar_Msun']=mstar[gi]
            rf_dict['SFR_Msunperyr']=sfr[gi]
            rf_dict['mergerNumber']=number[gi]
            
            rf_dict['gini']=gini[gi]
            rf_dict['m20']=m20[gi]

            if skip_mi is True:
                cols=['dGM20','fGM20','asym','Dstat','cc']
                rflabel='paramsmod2'
            else:
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
                result, labels, label_probability, rf_objs,roc_results,roc_tests = PyML.randomForestMC(newdf,iterations=RF_ITER,cols=cols,max_leaf_nodes=np.int32(Nmergers*0.5),max_features=3)    #,max_leaf_nodes=30,n_estimators=50
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
                np.save(arr=roc_tests,file='rfoutput/'+label+'/'+rflabel+'_roctests_{}_{}.npy'.format(sk,fk) )
                np.save(arr=rf_objs,file='rfoutput/'+label+'/'+rflabel+'_rfobj_{}_{}.npy'.format(sk,fk))



    
    sk='combined'
    fk='subset'
    print("Running Random Forest on combined dataset... ")
    result, labels, label_probability, rf_objs,roc_results,roc_tests = PyML.randomForestMC(full_df,iterations=RF_ITER,cols=cols)    #,max_leaf_nodes=30,n_estimators=50
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
    np.save(arr=roc_tests,file='rfoutput/'+label+'/'+rflabel+'_roctests_{}_{}.npy'.format(sk,fk) )  #.to_pickle('rfoutput/'+label+'/'+rflabel+'_roc_{}_{}.pkl'.format(sk,fk))

    np.save(arr=rf_objs,file='rfoutput/'+label+'/'+rflabel+'_rfobj_{}_{}.npy'.format(sk,fk))


                
    
    return


def load_all_candels(**kwargs):

    df1=pandas.DataFrame()
    df2=pandas.DataFrame()
    df3=pandas.DataFrame()
    
    fields=['cos','egs','gds-n','gds-s','uds']
    for f in fields:
        df1_f,df2_f,df3_f=load_candels_dfs(field=f,**kwargs)
        df1=df1.append(df1_f)
        df2=df2.append(df2_f)
        df3=df3.append(df3_f)
        
    return df1,df2,df3

def load_candels_dfs(field='egs',zrange=[1.75,2.25],mrange=[10.50,13.5],col_labels=['dGM20','fGM20','ASYM','MPRIME','I','D','CON','M']):
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
    mfi_mag=np.zeros_like(mfi_id,dtype=np.float64)
    
    mfj_mstar=np.zeros_like(mfj_id,dtype=np.float64)
    mfj_z=np.zeros_like(mfj_id,dtype=np.float64)
    mfj_mag=np.zeros_like(mfj_id,dtype=np.float64)

    mfh_mstar=np.zeros_like(mfh_id,dtype=np.float64)
    mfh_z=np.zeros_like(mfh_id,dtype=np.float64)
    mfh_mag=np.zeros_like(mfh_id,dtype=np.float64)
    
    
    for i,mfi_label in enumerate(mfi_id):
        ix= label==mfi_label
        if np.sum(ix)==1:
            mfi_mstar[i]=logm[ix]
            mfi_z[i]=z[ix]
        else:
            mfi_mstar[i]=0.0
            mfi_z[i]=-1.0
            
    for i,mfj_label in enumerate(mfj_id):
        ix= label==mfj_label
        if np.sum(ix)==1:
            mfj_mstar[i]=logm[ix]
            mfj_z[i]=z[ix]
        else:
            mfj_mstar[i]=0.0
            mfj_z[i]=-1.0
    for i,mfh_label in enumerate(mfh_id):
        ix= label==mfh_label
        if np.sum(ix)==1:
            mfh_mstar[i]=logm[ix]
            mfh_z[i]=z[ix]
        else:
            mfh_mstar[i]=0.0
            mfh_z[i]=-1.0
            
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


    snp_i=mfitab['SN_PIX_I']
    snp_j=mfjtab['SN_PIX_J']
    snp_h=mfhtab['SN_PIX_H']
    
    iix=(mfi_mstar >= mrange[0])*(mfi_mstar < mrange[1])*(mfi_z >= zrange[0])*(mfi_z < zrange[1]*(snp_i >= 3.0))
    jix=(mfj_mstar >= mrange[0])*(mfj_mstar < mrange[1])*(mfj_z >= zrange[0])*(mfj_z < zrange[1]*(snp_j >= 3.0))
    hix=(mfh_mstar >= mrange[0])*(mfh_mstar < mrange[1])*(mfh_z >= zrange[0])*(mfh_z < zrange[1]*(snp_h >= 3.0))

    
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
    dmi= dict_814['MPRIME']==-99.0
    dict_814['MPRIME'][dmi]=mfitab['M_I'][iix][dmi]
    
    dict_814['LMSTAR_BC03']=mfi_mstar[iix]
    dict_814['Z_BEST']=mfi_z[iix]
    dict_814['CANDELS_ID']=mfi_id[iix]
    dict_814['MAG_F814W']=mfi_mag[iix]
    dict_814['SNPIX_F814W']=snp_i[iix]
    
        
    dict_125={}
    dict_125[col_labels[0]]=mfj_1[jix]
    dict_125[col_labels[1]]=mfj_2[jix]
    dict_125[col_labels[2]]=mfjtab[col_labels[2]+'_J'][jix]
    dict_125[col_labels[3]]=mfjtab[col_labels[3]+'_J'][jix]
    dict_125[col_labels[4]]=mfjtab[col_labels[4]+'_J'][jix]
    dict_125[col_labels[5]]=mfjtab[col_labels[5]+'_J'][jix]
    dict_125[col_labels[6]]=mfjtab[col_labels[6]+'_J'][jix]
    dmi= dict_125['MPRIME']==-99.0
    dict_125['MPRIME'][dmi]=mfjtab['M_J'][jix][dmi]
    
    dict_125['LMSTAR_BC03']=mfj_mstar[jix]
    dict_125['Z_BEST']=mfj_z[jix]
    dict_125['CANDELS_ID']=mfj_id[jix]
    dict_125['MAG_F125W']=mfj_mag[jix]
    dict_125['SNPIX_F125W']=snp_j[jix]
    
    dict_160={}
    dict_160[col_labels[0]]=mfh_1[hix]
    dict_160[col_labels[1]]=mfh_2[hix]
    dict_160[col_labels[2]]=mfhtab[col_labels[2]+'_H'][hix]
    dict_160[col_labels[3]]=mfhtab[col_labels[3]+'_H'][hix]
    dict_160[col_labels[4]]=mfhtab[col_labels[4]+'_H'][hix]
    dict_160[col_labels[5]]=mfhtab[col_labels[5]+'_H'][hix]
    dict_160[col_labels[6]]=mfhtab[col_labels[6]+'_H'][hix]
    dmi= dict_160['MPRIME']==-99.0
    dict_160['MPRIME'][dmi]=mfhtab['M_H'][hix][dmi]
    
    dict_160['LMSTAR_BC03']=mfh_mstar[hix]
    dict_160['Z_BEST']=mfh_z[hix]
    dict_160['CANDELS_ID']=mfh_id[hix]
    dict_160['MAG_F160W']=mfh_mag[hix]
    dict_160['SNPIX_F160W']=snp_h[hix]
    
    df_814 = pandas.DataFrame(dict_814)
    df_125 = pandas.DataFrame(dict_125)
    df_160 = pandas.DataFrame(dict_160)
    
    df_814['field']=field
    df_125['field']=field
    df_160['field']=field
    
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
    axi2.set_ylim(0.3,3.1)
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
    elif rflabel=='paramsmod2':
        cols=['dGM20','fGM20','asym','Dstat','cc']
        syms=['^g','<r','ok','sy','*k']        
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
        roc_file= rfdata+rflabel+'_roctests_{}_{}.npy'.format(sk,fk)


        
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


        

        grid_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fk+'_gridplot.pdf'
        f6 = pyplot.figure(figsize=(6.5,6.5), dpi=300)
        pyplot.subplots_adjust(left=0.11, right=0.99, bottom=0.10, top=0.99,wspace=0.0,hspace=0.0)

        plot_rf_grid(f6,rf_data,rfprob,rf_class,rf_flag,cols)

        f6.savefig(grid_filen,dpi=300)
        pyplot.close(f6)     


        
    f1.savefig(plot_filen,dpi=300)
    pyplot.close(f1)

    axi2.legend(cols,fontsize=8,ncol=3,loc='upper center',numpoints=1)
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
            pyplot.legend(['nonmerger','merger',r'$P_{RF}>$'+'{:4.2f}'.format(rfthresh)],fontsize=sp+5,markerscale=4,scatterpoints=3,numpoints=3,framealpha=1.0,bbox_to_anchor=(0.35, 0.50),bbox_transform=pyplot.gcf().transFigure)
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






def make_merger_images(msF,merF,snap_key,fil_key,fil2_key=None,bd='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/',rflabel='paramsmod',rf_masscut=0.0,labelfunc='label_merger1',Npix=None,ckpc=75.0,ckpcz=None):

    j=-1
    sk=snap_key
    fku=fil_key
    fk=fku
    
    plotdir = 'images/'+labelfunc
    if fil2_key is None:
        plot_filen = plotdir+'/'+rflabel+'_mergers_'+sk+'_'+fku+'.pdf'
    else:
        plot_filen = plotdir+'/'+rflabel+'_mergers_'+sk+'_'+fku+'_'+fil2_key+'.pdf'
        
    if not os.path.lexists(plotdir):
        os.mkdir(plotdir)
        
    f1 = pyplot.figure(figsize=(12.0,10.0), dpi=600)
    pyplot.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,wspace=0.0,hspace=0.0)

    N_columns = 12
    N_rows = 10
    N_pix = Npix #im_npix[j]  #140
    
    

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

    if redshift > 4.2:
        return -1

    rfdata = 'rfoutput/'+labelfunc+'/'
        
    data_file = rfdata+rflabel+'_data_{}_{}.pkl'.format(sk,fku)
    result_file= rfdata+rflabel+'_importances_{}_{}.npy'.format(sk,fku)
    prob_file = rfdata+rflabel+'_probability_{}_{}.npy'.format(sk,fku)

    rf_data = np.load(data_file,encoding='bytes')
    rf_flag = rf_data['mergerFlag'].values
 

    #result = np.load(result_file)
    #completeness = np.median(result['completeness'].values)
    #ppv = np.median(result['ppv'].values)
        
    #probs = data frame with N random forest iterations, mergerFlag, and SubfindID
    probstuff = np.load(prob_file)
        
    #iters = 1#result['completeness'].values.shape[0]

    #probs_subframe = pandas.df(probs[0]) #probs[probs.keys()[0:iters]]

    #average_prob = probs_subframe.apply(np.mean,axis=1)
    average_prob=probstuff[:,0]
    flags = probstuff[:,1]
    rf_sfids = probstuff[:,2]

    im1_file=rf_data['IMFILE'].values
    if fil2_key is not None:
        im2_file=rf_data['IM2FILE'].values
    
    #print(flags.shape,gi.shape)


    #good_sfid = all_sfids[gi]
    #assert(np.all(good_sfid==rf_sfids)==True)
        
    #print('redshift: {:3.1f}   filter: {:15s}  # iterations:  {:8d}   RF sample size: {:8d}   # True mergers: {} '.format(redshift,fk,iters,rf_asym.shape[0],np.sum(rf_flag)))
    
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
                
            #im_i = gi[pi[pii]]  #gi indexes image array to RF inputs ; pi indexes this bin ; pii is this exact one
            #morph_sfid = all_sfids[gi[pi[pii]]]
            rf_sfid = rf_sfids[pi[pii]]
            #assert(morph_sfid==rf_sfid)
            
            #this_mer4 = boolean_merger4[gi[pi[pii]]]
            #this_match = mlpid_match[gi[pi[pii]]]

            if fil2_key is not None:
                r_im = bd+sk+'/'+im2_file[pi[pii]]
                g_im = bd+sk+'/'+im2_file[pi[pii]]
                b_im = bd+sk+'/'+im2_file[pi[pii]]
            else:
                r_im = bd+sk+'/'+im1_file[pi[pii]]
                g_im = bd+sk+'/'+im1_file[pi[pii]]
                b_im = bd+sk+'/'+im1_file[pi[pii]]
                
            rf_im = bd+sk+'/'+im1_file[pi[pii]]
            
            r = pyfits.open(r_im)[0].data
            g = pyfits.open(g_im)[0].data
            b = pyfits.open(b_im)[0].data
            
            pixsize_kpc=pyfits.open(g_im)[0].header['PIXKPC']
            
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

            r_header = pyfits.open(rf_im)[0].header
            this_camnum_int = r_header['CAMERA']
                
            this_prob = average_prob[pi[pii]]
                
            axi = f1.add_subplot(N_rows,N_columns,N_columns*(i+1)-k)
            axi.set_xticks([]) ; axi.set_yticks([])
            
            alph=0.2 ; Q=8.0

            print(b.shape,g.shape,r.shape)
            rgbthing = make_color_image.make_interactive(b,g,r,alph,Q)
            axi.imshow(rgbthing,interpolation='nearest',aspect='auto',origin='lower')


            #need to re-center and zoom subsequent plots
            #axi = overplot_morph_data(axi,rf_im,mid,delt)

                
            this_flag = flags[pi[pii]]
            #this_number= rf_number[pi[pii]]
            
            if this_flag==True:
                fcolor = 'Green'
            else:
                fcolor = 'Red'


            #if this_flag==True and this_match==True:
            #    axi.annotate('$unique$',(0.75,0.10),xycoords='axes fraction',ha='center',va='center',color=fcolor,size=8)
            #elif this_flag==True and this_match==False:
            #    axi.annotate('$infall$',(0.75,0.10),xycoords='axes fraction',ha='center',va='center',color=fcolor,size=8)
            if this_flag==True:
                axi.annotate('$True$',(0.75,0.10),xycoords='axes fraction',ha='center',va='center',color=fcolor,size=8)                    
            elif this_flag==False:
                axi.annotate('${:5s}$'.format(str(this_flag)),(0.75,0.10),xycoords='axes fraction',ha='center',va='center',color=fcolor,size=8)
                    
                    
            axi.annotate('${:4.2f}$'.format(this_prob),(0.25,0.10),xycoords='axes fraction',ha='center',va='center',color='White',size=7)
                        
            axi.annotate('${:7d}$'.format(rf_sfid),(0.25,0.90),xycoords='axes fraction',ha='center',va='center',color='White',size=6)

            axi.annotate('${:2d}$'.format(this_camnum_int),(0.25,0.82),xycoords='axes fraction',ha='center',va='center',color='White',size=3 )

            #axi.annotate('${:2d}$'.format(this_number),(0.75,0.86),xycoords='axes fraction',ha='center',va='center',color='DodgerBlue',size=3)
                
            #if this_flag==False and this_mer4==True:
            #    axi.annotate('$M$',(0.85,0.15),xycoords='axes fraction',ha='center',va='center',color='Green',size=6)
                    
                    
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






def plot_rpet_mass(msF,merF,sk,fk,FIG,Cval,gridf='median_grid',rpet=None,mstar=None,nx=4,ny=2,ii=1,i=1,redshift=0.0,skipthis=False,**bin_kwargs):

    if rpet is None:
        rpet  = get_all_morph_val(msF,sk,fk,'RP')
    if mstar is None:
        mstar  = get_all_snap_val(msF,sk,'Mstar_Msun')
    
    bins=18
    labs=10

    xlim=[9.7,11.9]
    ylim=[0.0,2.0]

    axi=FIG.add_subplot(ny,nx,ii)
    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])
        
    assert redshift > 0.0
    scale=illcos.kpc_proper_per_arcmin(redshift).value/60.0

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
        res,colorobj = gth.make_twod_grid(axi,np.log10(mstar[gi]),np.log10(rpet[gi]*scale),data_dict,gridf,bins=bins,**bin_kwargs)


    anny=0.92

    axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    axi.annotate(fk                             ,xy=(0.30,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)


    if i==2*nx:
        axi.set_ylabel('$log_{10}\ R_P [kpc]$',labelpad=1,size=labs)
    if i==nx:
        axi.set_xlabel('$log_{10}\ M_{*} [M_{\odot}]$',labelpad=1,size=labs)
    if i > nx:
        axi.set_xticklabels([])
    if i != nx and i !=2*nx:
        axi.set_yticklabels([])

    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])
    
    return colorobj




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




def plot_c_a(msF,merF,sk,fk,FIG,Cval,gridf='median_grid',c=None,a=None,nx=4,ny=2,ii=1,i=1,redshift=0.0,skipthis=False,**bin_kwargs):

    if c is None:
        c  = get_all_morph_val(msF,sk,fk,'CC')
    if a is None:
        a  = get_all_morph_val(msF,sk,fk,'ASYM')
    
    bins=18
    labs=10

    xlim=[0.0,1.0]
    ylim=[0.4,5.0]

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
        gi = np.arange(a.shape[0])
        data_dict = copy.copy(Cval)

    if skipthis is True:
        pass
        res=None
        colorobj=None
    else:
        res,colorobj = gth.make_twod_grid(axi,a[gi],c[gi],data_dict,gridf,bins=bins,**bin_kwargs)


    anny=0.92

    axi.annotate('$z = {:3.1f}$'.format(redshift) ,xy=(0.80,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    axi.annotate(fk                             ,xy=(0.30,anny),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)


    if i==2*nx:
        axi.set_ylabel('$C$',labelpad=1,size=labs)
    if i==nx:
        axi.set_xlabel('$A$',labelpad=1,size=labs)
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






def make_structure_plots(msF,merF,snap_keys_use,fil_keys_use,name='pc1',varname=None,bartitle='median PC1',vmin=-2,vmax=3,barticks=[-2,-1,0,1,2,3],
                         rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',rflabel='params',gridf='median_grid',log=False,rfthresh=0.4):
    
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


    plot3_filen = 'structure/'+labelfunc+'/C_A_'+name+'_evolution.pdf'

    f3 = pyplot.figure(figsize=(6.5,4.0), dpi=300)
    pyplot.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)

    
    plot4_filen = 'structure/'+labelfunc+'/rpet_mstar_'+name+'_evolution.pdf'

    f4 = pyplot.figure(figsize=(6.5,4.0), dpi=300)
    pyplot.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)

    

    nx=4
    ny=2
    
    i=0
    for sk,fk in zip(snap_keys_use,fil_keys_use):
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

        rpet = get_all_morph_val(msF,sk,fk,'RP')

        
        sfid = get_all_snap_val(msF,sk,'SubfindID')
        
        S_GM20 = SGM20(gini,m20)
        F_GM20 = FGM20(gini,m20)
        

        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        sfr = get_all_snap_val(msF,sk,'SFR_Msunperyr')

        log_mstar_mhalo = np.log10( mstar/mhalo )

        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
        

        rfdata = 'rfoutput/'+labelfunc+'/'
        
        data_file = rfdata+rflabel+'_data_{}_{}.pkl'.format(sk,fk)
        prob_file = rfdata+rflabel+'_probability_{}_{}.npy'.format(sk,fk)


        if (varname=='rfprob' or varname=='rf_flag' or varname=='rf_class'):
            if redshift > 4.2:
                skipthis=True
                sfr=None
                mstar=None
                rpet=rpet
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

                rf_rpet=rf_data['size'].values
                

                probs = np.load(prob_file)

                rfprob=probs[:,0]
                
                rf_class = rfprob > rfthresh

                #axi3.plot(roc['fpr'],roc['tpr'])

                sfr=rf_sfr
                mstar=rf_mstar
                rpet=rf_rpet
                g=rf_gini
                m20=rf_m20
                cc=rf_cc
                asym=rf_asym
        else:
            skipthis=False
            sfr=None
            mstar=None
            rpet=rpet
            g=gini
            m20=m20
            cc=cc
            asym=asym

        if log is True:
            arr=np.asarray( locals()[varname] )
            arr=np.float64(arr)
            
            Cval = np.log10( arr )
        else:            
            Cval = locals()[varname]
            
        ii=9-i*1

        colorobj=plot_sfr_mass(msF,merF,sk,fk,f1,Cval,vmin=vmin,vmax=vmax,sfr=sfr,mstar=mstar,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis)

        colorobj=plot_rpet_mass(msF,merF,sk,fk,f4,Cval,vmin=vmin,vmax=vmax,rpet=rpet,mstar=mstar,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis)
        
        colorobj=plot_g_m20(msF,merF,sk,fk,f2,Cval,vmin=vmin,vmax=vmax,g=g,m20=m20,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis)

        colorobj=plot_c_a(msF,merF,sk,fk,f3,Cval,vmin=vmin,vmax=vmax,c=cc,a=asym,gridf=gridf,ny=ny,nx=nx,ii=ii,i=i,redshift=redshift,skipthis=skipthis)

        
        if i==1:
            the_colorobj=copy.copy(colorobj)


    
    gth.make_colorbar(the_colorobj,f1,title=bartitle,ticks=barticks,loc=[0.12,0.65,0.17,0.03],format='%1d')

    gth.make_colorbar(the_colorobj,f2,title=bartitle,ticks=barticks,loc=[0.12,0.65,0.17,0.03],format='%1d')



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

    
def make_all_structures(msF,merF,sku,fku,rf_labelfunc,rflabel,rfthresh=0.4):

    res = make_structure_plots(msF,merF,sku,fku,name='pc1',bartitle='median PC1',vmin=-2,vmax=3,barticks=[-2,-1,0,1,2,3],labelfunc=rf_labelfunc,rflabel=rflabel,rfthresh=rfthresh)
    res = make_structure_plots(msF,merF,sku,fku,name='pc3',bartitle='median PC3',vmin=-1,vmax=2,barticks=[-1,0,1,2],labelfunc=rf_labelfunc,rflabel=rflabel,rfthresh=rfthresh)
                
    res = make_structure_plots(msF,merF,sku,fku,name='rfprob',bartitle='$log_{10} <P_{RF}>$',vmin=-2,vmax=0,barticks=[-2,-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=True,rflabel=rflabel,rfthresh=rfthresh)
                
    res = make_structure_plots(msF,merF,sku,fku,name='rf_flag',bartitle='$log_{10} f_{merger}$',gridf='log_fraction_grid',vmin=-2,vmax=0,barticks=[-2,-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel,rfthresh=rfthresh)
                
    res = make_structure_plots(msF,merF,sku,fku,name='rf_class',bartitle='$log_{10} f_{RF}(0.4)$',gridf='log_fraction_grid',vmin=-2,vmax=0,barticks=[-2,-1,0],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel,rfthresh=rfthresh)
                
    res = make_structure_plots(msF,merF,sku,fku,name='rf_prop',varname='rf_flag',bartitle='merger proportion',gridf='normed_proportion_grid',vmin=0,vmax=1,barticks=[0,1],
                               rf_masscut=10.0**(10.5),labelfunc=rf_labelfunc,log=False,rflabel=rflabel,rfthresh=rfthresh)
                

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


def apply_sim_rf_to_data(msF,merF,snap_key,fil_key,cols,fil2_key=None,dkz1=1.25,dkz2=1.75,rflabel='paramsmod',rf_masscut=0.0,labelfunc='label_merger1',rfthresh=0.4):

    sk=snap_key
    fku=fil_key
    f2ku=fil2_key
    
    rfdata = 'rfoutput/'+labelfunc+'/'

    #load RF object/results
    data_file = rfdata+rflabel+'_data_{}_{}.pkl'.format(sk,fku)
    result_file= rfdata+rflabel+'_importances_{}_{}.npy'.format(sk,fku)
    prob_file = rfdata+rflabel+'_probability_{}_{}.npy'.format(sk,fku)
    obj_file = rfdata+rflabel+'_rfobj_{}_{}.npy'.format(sk,fku)
    
    rf_data = np.load(data_file,encoding='bytes')
    
    rf_prob_arr = np.load(prob_file)
    rf_prob=rf_prob_arr[:,0]
    rf_flag = rf_prob_arr[:,1]
    
    rf_arr = np.load(obj_file)
    rf_container=rf_arr.all()
    rfo=rf_container.rfo
    print(rfo)

    rf_number = rf_data['mergerNumber']
    
    #load CANDELS data
    df1,df2,df3=load_all_candels(zrange=[dkz1,dkz2])  #f814, f125, f160


    #APPLY to data

    if rflabel=='twofilters_snp':
        #apply to df1 and df3.. need to x-match on CANDELS_ID
        mergedf=pandas.merge(df1,df3,on='CANDELS_ID')  #x=F814W, y=F160W
        print(mergedf.columns)
        print(cols)
        #['ASYM_x', 'CANDELS_ID', 'CON_x', 'D_x', 'I_x', 'LMSTAR_BC03_x',
       #'MAG_F814W', 'MPRIME_x', 'Z_BEST_x', 'dGM20_x', 'fGM20_x', 'field_x',
       #'ASYM_y', 'CON_y', 'D_y', 'I_y', 'LMSTAR_BC03_y', 'MAG_F160W',
       #'MPRIME_y', 'Z_BEST_y', 'dGM20_y', 'fGM20_y', 'field_y']
       
        #['dGM20', 'fGM20', 'asym', 'Dstat', 'cc', 'snp','dGM20_2', 'fGM20_2', 'asym_2', 'Dstat_2', 'cc_2']
        datacols=['dGM20_x','fGM20_x','ASYM_x','D_x','CON_x','SNPIX_F814W','dGM20_y','fGM20_y','ASYM_y','D_y','CON_y','SNPIX_F160W']
    elif rflabel=='twofilters':
        mergedf=pandas.merge(df1,df3,on='CANDELS_ID')  #x=F814W, y=F160W
        datacols=['dGM20_x','fGM20_x','ASYM_x','D_x','CON_x','dGM20_y','fGM20_y','ASYM_y','D_y','CON_y']
     
        
    #are CON values whack?
    diff1=np.median(rf_data['cc'])-np.median(mergedf['CON_x'].values)
    diff2=np.median(rf_data['cc_2'])-np.median(mergedf['CON_y'].values)

    #were these values already fixed?
    #mergedf['CON_x']=mergedf['CON_x']+diff1
    #mergedf['CON_y']=mergedf['CON_y']+diff2
        
    print(np.median( rf_data['cc']), np.median(mergedf['CON_x'].values))
    print(np.median( rf_data['cc_2']), np.median(mergedf['CON_y'].values))

    candels_values=mergedf[datacols].values
        
    candels_df=mergedf


        
    thresh=rfthresh
        
    candels_prob=rfo.predict_proba(candels_values)[:,1]
    candels_df['RF_PROB']=candels_prob
    
    rf_total=rf_prob.shape[0]
    rf_class=np.sum(rf_prob >= thresh)
    rf_real =np.sum(rf_flag==True)

    mergerspertrue=np.sum(rf_number[rf_flag==True])/float( np.sum(rf_flag==True))

    
    candels_total=candels_prob.shape[0]
    candels_class=np.sum(candels_prob >= thresh)
    candels_asym=np.sum(candels_df['ASYM_x'] > 0.25) 
    
    print(mergerspertrue, candels_asym, candels_class, candels_total)
    print('tru merger   fraction: {:5.3f}'.format( float(rf_real)/rf_total ))
    print('RF  merger   fraction: {:5.3f}'.format( float(rf_class)/rf_total ))
    print('HST rfclass  fraction: {:5.3f}'.format( float(candels_class)/candels_total ))
    print('HST asym     fraction: {:5.3f}'.format( float(candels_asym)/candels_total ))


    ncol=len(datacols)
    grid_filen = 'rf_plots/'+labelfunc+'/'+rflabel+'_'+sk+'_'+fku+'_candelsgrid.pdf'
    f6 = pyplot.figure(figsize=(ncol+0.5,ncol+0.5), dpi=300)
    pyplot.subplots_adjust(left=0.11, right=0.99, bottom=0.10, top=0.99,wspace=0.0,hspace=0.0)

    plot_rf_grid(f6,candels_df,candels_prob,None,None,datacols)
    
    f6.savefig(grid_filen,dpi=300)
    pyplot.close(f6)     


    
        
    #save results in same folder as above *dataprobs* ?
    candels_df.to_pickle(rfdata+rflabel+'_candelsclass_{}_{}.npy'.format(sk,fku))

    
    return



def print_tables(msF,merF,fil_keys,rf_labelfuncs=['label_merger_window500_both','label_merger_forward250_both'],rflabel='paramsmod2',rf_masscut=10.0**(10.5)):

    print(r'\begin{table*}')
    print(r'\centering')
    print(r'\caption{Dataset Properties}')
    print(r'\label{tab:dataset}')
    print(r'\begin{tabular}{ccccccc}')
    print(r'Snapshot & z & $N_{gal}$ & $N_{\rm mergers}$ & N images &  $N_{\rm mergers}^{500}$  &  $N_{\rm mergers}^{250}$  \\')
    print(r' & & \multicolumn{2}{c}{$ $} & \multicolumn{3}{c}{$ M_* > 10^{10.5} M_{\odot}$}  \\')

    snapshots=['103','085','075','068','064','060','054','049','045','041','038','035']
    for sn in snapshots:
        sk='snapshot_'+sn

        mstar_cam0=get_all_snap_val(msF,sk,'Mstar_Msun',camera='CAMERA0')
        mstar_all=get_all_snap_val(msF,sk,'Mstar_Msun')
        
        gi= mstar_cam0 >= rf_masscut
        gia= mstar_all >= rf_masscut
        
        redshift = msF['nonparmorphs'][sk]['NC-F200W']['CAMERA0']['REDSHIFT'].value[0]

        boolean_flag1,number1 = eval(rf_labelfuncs[0]+'(merF,sk)')
        boolean_flag2,number2 = eval(rf_labelfuncs[1]+'(merF,sk)')
        #print(np.sum(boolean_flag1))
        
        
        print(r'{:5s} & {:5.1f} & {:8d} & {:8d} & {:8d} & {:8d} & {:8d} \\'.format(sn,redshift,mstar_cam0.shape[0],mstar_cam0[gi].shape[0],mstar_all[gia].shape[0],np.sum(boolean_flag1[gia]),np.sum(boolean_flag2[gia])))

    print(r'\end{tabular}')
    print(r'\end{table*}')



def do_snapshot_evolution(msF,merF,
                          sku =['snapshot_103','snapshot_085','snapshot_075','snapshot_068','snapshot_064','snapshot_060','snapshot_054','snapshot_049'],
                          fku =['ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ],
                          f2ku=['WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ],
                          labelfunc='label_merger_window500_both',skipcalc=False,rfthresh=0.2):


    for sk,fk,fk2 in zip(sku,fku,f2ku):
    
        rflabel,cols=simple_random_forest(msF,merF,sk,fk, paramsetting='onefilter',max_leaf_nodes=25,
                                          rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,thresh=rfthresh)
        rflabel,cols=simple_random_forest(msF,merF,sk,fk2, paramsetting='onefilter',max_leaf_nodes=25,
                                          rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,thresh=rfthresh)
        
        rflabel,cols=simple_random_forest(msF,merF,sk,fk,fil2_key=fk2, paramsetting='twofilters',max_leaf_nodes=25,
                                          rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,thresh=rfthresh)


        rflabel,cols=simple_random_forest(msF,merF,sk,fk, paramsetting='onefilter_snp',max_leaf_nodes=25,
                                          rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,thresh=rfthresh)
        rflabel,cols=simple_random_forest(msF,merF,sk,fk2, paramsetting='onefilter_snp',max_leaf_nodes=25,
                                          rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,thresh=rfthresh)
        
        rflabel,cols=simple_random_forest(msF,merF,sk,fk,fil2_key=fk2, paramsetting='twofilters_snp',max_leaf_nodes=25,
                                          rf_masscut=10.0**(10.5),labelfunc=labelfunc,skipcalc=skipcalc,thresh=rfthresh)
        
        apply_sim_rf_to_data(msF,merF,sk,fk,cols,fil2_key=fk2,rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc=labelfunc,dkz1=1.25,dkz2=1.75,rfthresh=rfthresh)

        res = make_all_structures(msF,merF,sku,fku,rf_labelfunc=labelfunc,rflabel=rflabel,rfthresh=rfthresh)

        #res = make_merger_images(msF,merF,sk,fk,rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc=labelfunc)
        #res = make_merger_images(msF,merF,sk,fk,fil2_key=fk2,rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc=labelfunc)
        

    
        
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

                
                print_tables(msF,merF,fil_keys_use,rf_labelfuncs=['label_merger_window500_both','label_merger_forward250_both'],rflabel=rflabel)
                #exit()


                #simple_random_forest(msF,merF,'snapshot_075','WFC3-F160W', paramsetting='medium',rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both')
                #simple_random_forest(msF,merF,'snapshot_075','WFC3-F160W', paramsetting='shapeonly',rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both')


                '''
                rflabel,cols=simple_random_forest(msF,merF,'snapshot_075','WFC3-F160W', paramsetting='onefilter',max_leaf_nodes=25,
                                                  rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',skipcalc=False)
                rflabel,cols=simple_random_forest(msF,merF,'snapshot_068','WFC3-F160W', paramsetting='onefilter',max_leaf_nodes=25,
                                                  rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',skipcalc=False)
                
                rflabel,cols=simple_random_forest(msF,merF,'snapshot_075','ACS-F814W',fil2_key='WFC3-F160W', paramsetting='medium',max_leaf_nodes=25,
                                                  rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',skipcalc=False)
                rflabel,cols=simple_random_forest(msF,merF,'snapshot_068','ACS-F814W',fil2_key='WFC3-F160W', paramsetting='medium',max_leaf_nodes=25,
                                                  rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',skipcalc=False)
                '''
                
                #res = make_merger_images(msF,merF,'snapshot_075','ACS-F814W',rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both')
                #res = make_merger_images(msF,merF,'snapshot_075','ACS-F814W',fil2_key='WFC3-F160W',rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both')

                #apply_sim_rf_to_data(msF,merF,'snapshot_075','ACS-F814W',cols,fil2_key='WFC3-F160W',rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',dkz1=1.25,dkz2=1.75)
                
                #localvars = run_random_forest(msF,merF,snap_keys,fil_keys_use,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',balancetrain=False,skip_mi=skipmi)
                #localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys_use),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both',twin=0.5)
                #res = make_all_structures(msF,merF,rf_labelfunc='label_merger_window500_both',rflabel=rflabel)

                
                #localvars = run_random_forest(msF,merF,snap_keys,fil_keys_use,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_forward250_both',balancetrain=False,skip_mi=skipmi)
                #localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys_use),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_forward250_both',twin=0.25)
                #res = make_all_structures(msF,merF,rf_labelfunc='label_merger_forward250_both',rflabel=rflabel)



                #something like this:
                '''
                do_snapshot_evolution(msF,merF,
                                      sku =['snapshot_103','snapshot_085','snapshot_075','snapshot_068','snapshot_064','snapshot_060','snapshot_054','snapshot_049'],
                                      fku =['ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ,'ACS-F814W'   ],
                                      f2ku=['WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ,'WFC3-F160W'  ],
                                      labelfunc='label_merger_window500_both',skipcalc=False)
                '''
                do_snapshot_evolution(msF,merF,
                                      sku =['snapshot_075'],
                                      fku =['ACS-F814W'],
                                      f2ku=['WFC3-F160W'],
                                      labelfunc='label_merger_window500_both',skipcalc=False)
                
                #do_snapshot_evolution(msF,merF,
                #                      sku =['snapshot_075'],
                #                      fku =['WFC3-F160W'],
                #                      f2ku=[None],
                #                      labelfunc='label_merger_window500_both',skipcalc=False)                
                '''
                localvars = run_random_forest(msF,merF,snap_keys,fil_keys_use,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_past250_both',balancetrain=False)
                localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys_use),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_past250_both',twin=0.25)
                res = make_all_structures(msF,merF,rf_labelfunc='label_merger_past250_both',rflabel=rflabel)
                '''
                

                #don't have merF125 that we need for this one
                #localvars = run_random_forest(msF,merF,snap_keys,fil_keys,rfiter=5,rf_masscut=10.0**(10.5),labelfunc='label_merger_window250_both',balancetrain=False)
                #localvars = make_rf_evolution_plots(copy.copy(snap_keys),copy.copy(fil_keys),rflabel=rflabel,rf_masscut=10.0**(10.5),labelfunc='label_merger_window250_both',twin=0.25)
                #res = make_all_structures(msF,merF,rf_labelfunc='label_merger_window250_both',rflabel=rflabel)


                

                #res = make_merger_images(msF,merF,fil_keys_use,rflabel='params',rf_masscut=10.0**(10.5),labelfunc='label_merger_window500_both')
                #res = make_merger_images(msF,merF,fil_keys_use,rflabel='params',rf_masscut=10.0**(10.5),labelfunc='label_merger_forward250_both')
                
                
                
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

                


                

                #res=do_rf_result_grid(copy.copy(snap_keys),copy.copy(fil_keys_use),rflabel=rflabel,rf_labelfunc='label_merger_window500_both')




                
