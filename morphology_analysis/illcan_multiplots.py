import math
import string
import sys
import struct
import matplotlib
matplotlib.use('Agg')
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
from scipy.integrate import quad
import glob
import os
import gzip
import shutil
import congrid
import astropy.io.ascii as ascii
import warnings
import subprocess
import astropy
import astropy.io.fits as pyfits
import astropy.units as u
from astropy.cosmology import WMAP7,z_at_value
import copy
import datetime
import ezgal
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
import parse_illustris_morphs as pim
import illustris_python as ilpy
from parse_illustris_morphs import *
from PyML import machinelearning as pyml
from PyML import convexhullclassifier as cvx
import statmorph
import gfs_twod_histograms as gth
import pandas

from PyML import machinelearning as pyml


ilh = 0.704
illcos = astropy.cosmology.FlatLambdaCDM(H0=70.4,Om0=0.2726,Ob0=0.0456)
sq_arcsec_per_sr = 42545170296.0



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



def fGM20(gini_good,m20_good):
    Fgm20_good=[]
    F=(-0.693*m20_good)+(4.95*gini_good)-3.85
    if gini_good[i]>= ((0.14*m20_good[i])+0.778):
	Fgm20_good.append(abs(F))
    if gini_good[i]< ((0.14*m20_good[i])+0.778):
        Fgm20_good.append(-abs(F))

    return

def dGM20(gini_good,m20_good):
    Dgm20_good = []
    Dgm20_good.append(abs((-0.14*m20_good[i])-gini_good[i]+0.33)/(0.14))
        
    return Dgm20_good
    
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

    npmorph = pyml.dataMatrix(pcd,parameters)
    pc = pyml.pcV(npmorph)

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


    gi = np.where(np.isfinite(Cval)==True)[0]
    
    res = gth.make_twod_grid(axi,np.log10(mstar[gi]),np.log10(sfr[gi]),{'x':Cval[gi]},gridf,**bin_kwargs)
    


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

    axi,colorobj = gth.make_twod_grid(axi,np.log10(mstar[gi]),np.log10(size_kpc[gi]),{'x':Cval[gi]},gridf,**bin_kwargs)
    axi.annotate('z = {:3.1f}'.format(redshift) ,xy=(0.80,0.05),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    axi.annotate(fk                             ,xy=(0.80,0.15),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)

    return colorobj



def make_sfr_radius_mass_plots(msF,merF):

    snap_keys = ['snapshot_103','snapshot_085','snapshot_075','snapshot_075','snapshot_068','snapshot_068',
                 'snapshot_064','snapshot_064','snapshot_060','snapshot_060','snapshot_054', 'snapshot_049','snapshot_041']
    fil_keys =  ['ACS-F606W',   'ACS-F814W',  'WFC3-F105W',  'NC-F115W',    'WFC3-F160W',  'NC-F115W',
                 'WFC3-F160W',  'NC-F150W'    ,'WFC3-F160W',  'NC-F200W'    ,'NC-F200W'    , 'NC-F277W',   'NC-F277W']
    
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

        this_NumMajorMergersLastGyr = get_mergerinfo_val(merF,sk,'this_NumMajorMergersLastGyr')
        boolean_merger2 = this_NumMajorMergersLastGyr >= 1.0
        

        mhalo = get_all_snap_val(msF,sk,'Mhalo_Msun')
        mstar = get_all_snap_val(msF,sk,'Mstar_Msun')
        log_mstar_mhalo = np.log10( mstar/mhalo )

        redshift = msF['nonparmorphs'][sk][fk]['CAMERA0']['REDSHIFT'].value[0]
        
        #set up RF data frame above, run or save input/output for each loop iteration

        rf_dict = {}
        PARAMS_MOD=True
        PARAMS_ONLY=False
        PCS_ONLY=False
        RUN_RF=False
        RF_ITER=3
        
        if PCS_ONLY is True:
            gi = np.where(np.isfinite(pc1)*np.isfinite(pc2)*np.isfinite(pc3)*np.isfinite(pc4)*np.isfinite(pc5)*np.isfinite(pc6)*np.isfinite(pc7) != 0)[0]
            print gi.shape, pc1.shape
            rf_dict['pc1']=pc1[gi]
            rf_dict['pc2']=pc2[gi]
            rf_dict['pc3']=pc3[gi]
            rf_dict['pc4']=pc4[gi]
            rf_dict['pc5']=pc5[gi]
            rf_dict['pc6']=pc6[gi]
            rf_dict['pc7']=pc7[gi]
            rf_dict['mergerFlag']=boolean_merger1[gi]
            cols=['pc1','pc2','pc3','pc4','pc5','pc6','pc7']
            rflabel='pcs'
            
        if PARAMS_ONLY is True:
            gi = np.where(np.isfinite(gini)*np.isfinite(m20)*np.isfinite(asym)*np.isfinite(Mstat)*np.isfinite(Istat)*np.isfinite(Dstat)*np.isfinite(cc) != 0)[0]
            print gi.shape, pc1.shape
            rf_dict['gini']=gini[gi]
            rf_dict['m20']=m20[gi]
            rf_dict['asym']=asym[gi]
            rf_dict['Mstat']=Mstat[gi]
            rf_dict['Istat']=Istat[gi]
            rf_dict['Dstat']=Dstat[gi]
            rf_dict['cc']=cc[gi]
            rf_dict['mergerFlag']=boolean_merger1[gi]
            cols=['gini','m20','asym','Mstat','Istat','Dstat','cc']
            rflabel='params'
            
        if PARAMS_MOD is True:
            gi = np.where(np.isfinite(S_GM20)*np.isfinite(F_GM20)*np.isfinite(asym)*np.isfinite(Mstat)*np.isfinite(Istat)*np.isfinite(Dstat)*np.isfinite(cc) != 0)[0]
            print gi.shape, pc1.shape
            rf_dict['dGM20']=S_GM20[gi]
            rf_dict['fGM20']=F_GM20[gi]
            rf_dict['asym']=asym[gi]
            rf_dict['Mstat']=Mstat[gi]
            rf_dict['Istat']=Istat[gi]
            rf_dict['Dstat']=Dstat[gi]
            rf_dict['cc']=cc[gi]
            rf_dict['mergerFlag']=boolean_merger1[gi]
            cols=['dGM20','fGM20','asym','Mstat','Istat','Dstat','cc']
            rflabel='paramsmod'

            
        if RUN_RF is True:
            if redshift < 4.2:
            
                df=pandas.DataFrame(rf_dict)
            
                print "Running Random Forest... ", sk, fk
                result, labels, label_probability = pyml.randomForestMC(df,iterations=RF_ITER,cols=cols)
                #result = summary statistics, feature importances (N iterations x N statistics/importances)
                #labels = labels following random forest (N galaxies x N iterations)
                #label_probability = probability of label following random forest (N galaxies x N iterations)

                #saves the output as a file
                if not os.path.lexists('rfoutput'):
                    os.mkdir('rfoutput')


                labels['mergerFlag']=df['mergerFlag']
                label_probability['mergerFlag']=df['mergerFlag']
                
                df.to_pickle('rfoutput/'+rflabel+'_data_cut_{}_{}.pkl'.format(sk,fk))
                result.to_pickle('rfoutput/'+rflabel+'_result_cut_{}_{}.pkl'.format(sk,fk))
                labels.to_pickle('rfoutput/'+rflabel+'_labels_cut_{}_{}.pkl'.format(sk,fk))
                label_probability.to_pickle('rfoutput/'+rflabel+'_label_probability_cut_{}_{}.pkl'.format(sk,fk))
                PCs.to_pickle('rfoutput/'+rflabel+'_pc_cut_{}_{}.pkl'.format(sk,fk))


        
        bins=18

        xlim=[9.7,12.2]
        ylim=[-2.0,3.0]
        rlim=[0.1,1.7]
        
        
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

        
        
    return locals()





def plot_g_m20_cc_a(msF,merF,sk,fk,FIG,xlim,x2lim,ylim,y2lim,Cval,gridf='median_grid',**bin_kwargs):

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
    axi.tick_params(axis='both',which='both',labelsize=labs)
    axi.locator_params(nbins=5,prune='both')

    axi.set_xlim(xlim[0],xlim[1])
    axi.set_ylim(ylim[0],ylim[1])


    gi = np.where(np.isfinite(Cval)==True)[0]
    
    res = gth.make_twod_grid(axi,asym,cc,{'x':Cval[gi]},gridf,**bin_kwargs)
    


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

    axi,colorobj = gth.make_twod_grid(axi,m20,gini,{'x':Cval[gi]},gridf,**bin_kwargs)
    axi.annotate('z = {:3.1f}'.format(redshift) ,xy=(0.80,0.05),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)
    axi.annotate(fk                             ,xy=(0.80,0.15),ha='center',va='center',xycoords='axes fraction',color='black',size=labs)

    return colorobj




def make_morphology_plots(msF,merF,vardict):

    for sk,fk in zip(vardict['snap_keys'],vardict['fil_keys']):

        bins=18

        xlim=[0.0,0.5]
        x2lim=[-0.8,-3.5]
        ylim=[0.4,5.0]
        y2lim=[0.35,0.75]
        
        
        plot_filen = 'pc1/morphology_'+sk+'_'+fk+'_pc1.pdf'
        if not os.path.lexists('pc1'):
            os.mkdir('pc1')
        
        f1 = pyplot.figure(figsize=(3.5,5.0), dpi=300)
        pyplot.subplots_adjust(left=0.15, right=0.98, bottom=0.08, top=0.88,wspace=0.0,hspace=0.10)
        colorobj = plot_g_m20_cc_a(msF,merF,sk,fk,f1,xlim=xlim,x2lim=x2lim,ylim=ylim,y2lim=y2lim,Cval=vardict['pc1'],vmin=-2,vmax=3,bins=bins)
        gth.make_colorbar(colorobj,title='PC1 morphology',ticks=[-2,-1,0,1,2,3])

        
        f1.savefig(plot_filen,dpi=300)
        pyplot.close(f1)

    
    return locals()




if __name__=="__main__":


    morph_stat_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/nonparmorphs_SB25_12filters_all_NEW.hdf5'
    merger_file = '/astro/snyder_lab2/Illustris/MorphologyAnalysis/imagedata_mergerinfo.hdf5'


    with h5py.File(morph_stat_file,'r') as msF:
        with h5py.File(merger_file,'r') as merF:
            localvars = make_sfr_radius_mass_plots(msF,merF)
            localvars = make_morphology_plots(msF,merF,localvars)
    
