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


ilh = 0.704
illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)
sq_arcsec_per_sr = 42545170296.0


snap_keys = ['snapshot_103','snapshot_103','snapshot_085','snapshot_085','snapshot_075','snapshot_075','snapshot_075','snapshot_068','snapshot_068','snapshot_068',
             'snapshot_064','snapshot_064','snapshot_060','snapshot_060','snapshot_054', 'snapshot_049','snapshot_041']
fil_keys =  ['ACS-F606W',  'WFC3-F105W',   'ACS-F814W',  'WFC3-F105W',  'WFC3-F105W',  'NC-F115W',  'NC-F150W',    'WFC3-F160W',  'NC-F115W',  'NC-F150W',
             'WFC3-F160W',  'NC-F150W'    ,'WFC3-F160W',  'NC-F200W'    ,'NC-F200W'    , 'NC-F277W',   'NC-F277W']    

im_snap_keys = ['snapshot_103','snapshot_085','snapshot_075','snapshot_068',
             'snapshot_064','snapshot_060','snapshot_054', 'snapshot_049']

im_rf_fil_keys = ['ACS-F606W','ACS-F814W','NC-F115W','NC-F115W',
                  'NC-F150W','NC-F200W','NC-F200W','NC-F277W']

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


def get_all_morph_val(msF,sk,fk,keyname):
    morph_array = np.concatenate( (msF['nonparmorphs'][sk][fk]['CAMERA0'][keyname].value,
                                   msF['nonparmorphs'][sk][fk]['CAMERA1'][keyname].value,
                                   msF['nonparmorphs'][sk][fk]['CAMERA2'][keyname].value,
                                   msF['nonparmorphs'][sk][fk]['CAMERA3'][keyname].value) )
    return morph_array


def get_all_snap_val(msF,sk,keyname):
    val_array = np.concatenate( (msF['nonparmorphs'][sk][keyname].value,
                                   msF['nonparmorphs'][sk][keyname].value,
                                   msF['nonparmorphs'][sk][keyname].value,
                                   msF['nonparmorphs'][sk][keyname].value) )
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

def label_merger4(merF,sk):
    latest_NumMinorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMinorMergersLastGyr')
    label_boolean = latest_NumMinorMergersLastGyr >= 1.0
    return label_boolean

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





def run_random_forest(msF,merF,rfiter=3,RUN_RF=True,rf_masscut=10.0**(10.5),labelfunc='label_merger1'):
    
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
        
        
        #latest_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'latest_NumMajorMergersLastGyr')
        #boolean_merger1 = latest_NumMajorMergersLastGyr >= 1.0
        boolean_flag = eval(labelfunc+'(merF,sk)')

        
        #this_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLastGyr')
        #boolean_merger2 = this_NumMajorMergersLastGyr >= 1.0
        

        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        log_mstar_mhalo = np.log10( mstar/mhalo )

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
            gi = np.where(np.isfinite(gini)*np.isfinite(m20)*np.isfinite(asym)*np.isfinite(Mstat)*np.isfinite(Istat)*np.isfinite(Dstat)*np.isfinite(cc)*(mstar >= rf_masscut) != 0)[0]
            print(gi.shape, pc1.shape)
            rf_dict['gini']=gini[gi]
            rf_dict['m20']=m20[gi]
            rf_dict['asym']=asym[gi]
            rf_dict['Mstat']=Mstat[gi]
            rf_dict['Istat']=Istat[gi]
            rf_dict['Dstat']=Dstat[gi]
            rf_dict['cc']=cc[gi]
            rf_dict['mergerFlag']=boolean_flag[gi]
            rf_dict['SubfindID']=sfid[gi]

            cols=['gini','m20','asym','Mstat','Istat','Dstat','cc']
            rflabel='params'
            label=labelfunc

            
        if PARAMS_MOD is True:
            gi = np.where(np.isfinite(S_GM20)*np.isfinite(F_GM20)*np.isfinite(asym)*np.isfinite(Mstat)*np.isfinite(Istat)*np.isfinite(Dstat)*np.isfinite(cc)*(mstar >= rf_masscut) != 0)[0]
            print(gi.shape, pc1.shape)
            rf_dict['dGM20']=S_GM20[gi]
            rf_dict['fGM20']=F_GM20[gi]
            rf_dict['asym']=asym[gi]
            rf_dict['Mstat']=Mstat[gi]
            rf_dict['Istat']=Istat[gi]
            rf_dict['Dstat']=Dstat[gi]
            rf_dict['cc']=cc[gi]
            rf_dict['mergerFlag']=boolean_flag[gi]
            rf_dict['SubfindID']=sfid[gi]

            cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc']
            rflabel='paramsmod'
            label=labelfunc

            
        if RUN_RF is True:
            if redshift < 4.2:
            
                df=pandas.DataFrame(rf_dict)
            
                print("Running Random Forest... ", sk, fk)
                result, labels, label_probability = PyML.randomForestMC(df,iterations=RF_ITER,cols=cols)
                #result = summary statistics, feature importances (N iterations x N statistics/importances)
                #labels = labels following random forest (N galaxies x N iterations)
                #label_probability = probability of label following random forest (N galaxies x N iterations)

                #saves the output as a file
                if not os.path.lexists('rfoutput'):
                    os.mkdir('rfoutput')
                if not os.path.lexists('rfoutput/'+label):
                    os.mkdir('rfoutput/'+label)

                labels['mergerFlag']=df['mergerFlag']
                label_probability['mergerFlag']=df['mergerFlag']
                labels['SubfindID']=df['SubfindID']
                label_probability['SubfindID']=df['SubfindID']

                
                df.to_pickle('rfoutput/'+label+'/'+rflabel+'_data_cut_{}_{}.pkl'.format(sk,fk))
                result.to_pickle('rfoutput/'+label+'/'+rflabel+'_result_cut_{}_{}.pkl'.format(sk,fk))
                labels.to_pickle('rfoutput/'+label+'/'+rflabel+'_labels_cut_{}_{}.pkl'.format(sk,fk))
                label_probability.to_pickle('rfoutput/'+label+'/'+rflabel+'_label_probability_cut_{}_{}.pkl'.format(sk,fk))
                PCs.to_pickle('rfoutput/'+label+'/'+rflabel+'_pc_cut_{}_{}.pkl'.format(sk,fk))

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


    gi = np.where( (np.isfinite(Cval)==True)*(mstar >= masscut))[0]
    
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


def make_rf_evolution_plots(rflabel='paramsmod',labelfunc='label_merger1',rf_masscut=0.0):


    plot_filen = 'rf_plots/global_stats.pdf'
    if not os.path.lexists('rf_plots'):
        os.mkdir('rf_plots')
    f1 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)


    imp_filen = 'rf_plots/importance_stats.pdf'
    if not os.path.lexists('rf_plots'):
        os.mkdir('rf_plots')
    f2 = pyplot.figure(figsize=(3.5,3.0), dpi=300)
    pyplot.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.98,wspace=0.0,hspace=0.0)


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
    
    imp_norm = 1.0/7.0

    if rflabel=='paramsmod':
        cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc']
        syms=['^g','<r','ok','sb','sg','sy','*k']
    else:
        assert(False)


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
        probs = np.load(prob_file)
        iters = result['completeness'].values.shape[0]

        print('redshift: {:3.1f}   filter: {:15s}   RF sample size: {:8d}   # True mergers: {} '.format(redshift,fk,rf_asym.shape[0],np.sum(rf_flag)))
        

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

        
    f1.savefig(plot_filen,dpi=300)
    pyplot.close(f1)

    axi2.legend(cols,fontsize=8,ncol=3,loc='upper center',numpoints=1)
    f2.savefig(imp_filen,dpi=300)
    pyplot.close(f2)

        
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


if __name__=="__main__":


    #morph_stat_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB25_12filters_all_NEW.hdf5'
    morph_stat_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB25_12filters_all_FILES.hdf5'

    #morph_stat_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB27_12filters_all_NEW.hdf5'

    #merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo.hdf5'
    #merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo_SB27.hdf5'
    merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo_SB25.hdf5'


    with h5py.File(morph_stat_file,'r') as msF:
        with h5py.File(merger_file,'r') as merF:
            localvars = make_sfr_radius_mass_plots(msF,merF,rfiter=10)
            
            #localvars = make_morphology_plots(msF,merF)
            #res = make_pc1_images(msF,merF,bd='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/')
            
            #localvars = run_random_forest(msF,merF,rfiter=3,rf_masscut=10.0**(10.5),labelfunc='label_merger4')
            
            #res = make_merger_images(msF,merF,rflabel='paramsmod',rf_masscut=10.0**(10.5),labelfunc='label_merger4')
            #localvars = make_rf_evolution_plots(rflabel='paramsmod',rf_masscut=10.0**(10.5),labelfunc='label_merger4')

