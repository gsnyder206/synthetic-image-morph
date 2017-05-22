import astropy.io.fits as pyfits
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.ndimage
import scipy as sp

import make_color_image
import illcan_multiplots as icmp
import glob
import gfs_sublink_utils as gsu

def showgalaxy(axi,snapkey,subfindID,camera,filters=['NC-F115W','NC-F150W','NC-F200W'],alph=0.2,Q=8.0,Npix=400,ckpc=None,ckpcz=None,sb='SB25',rfkey=None,dx=0,dy=0):
    image_base='/astro/snyder_lab/Illustris_CANDELS/Illustris-1_z1_images_bc03/'+snapkey
    
    camstr = camera
    b_fn = np.asarray(glob.glob(image_base+'/subdir_???/images_subhalo_'+str(subfindID)+'/*'+'cam'+camstr+'_'+filters[0]+'_'+sb+'.fits' ))
    g_fn = np.asarray(glob.glob(image_base+'/subdir_???/images_subhalo_'+str(subfindID)+'/*'+'cam'+camstr+'_'+filters[1]+'_'+sb+'.fits' ))
    r_fn = np.asarray(glob.glob(image_base+'/subdir_???/images_subhalo_'+str(subfindID)+'/*'+'cam'+camstr+'_'+filters[2]+'_'+sb+'.fits' ))

    
    #print('Accessing...',b_fn,g_fn,r_fn)

    try:
        r = pyfits.open(r_fn[0])[0].data
        g = pyfits.open(g_fn[0])[0].data
        b = pyfits.open(b_fn[0])[0].data
        pixsize_arcsec=pyfits.open(g_fn[0])[0].header['PIXSCALE']
        pixsize_kpc=pyfits.open(g_fn[0])[0].header['PIXKPC']
        
    except:
        print(r_fn,g_fn,b_fn)
        print('Could not open files requested.')
        return axi
        
    mid = np.int64(r.shape[0]/2)
    if Npix is not None:
        delt=np.int64(Npix/2)
    elif ckpc is not None:
        snap_int=np.int32(snapkey[-3:])
        this_z=gsu.redshift_from_snapshot(snap_int)
        #factor=icmp.illcos.kpc_comoving_per_arcmin(this_z).value/60.0
        
        Npix=ckpc/pixsize_kpc
        
        delt=np.int64(Npix/2)
    elif ckpcz is not None:
        snap_int=np.int32(snapkey[-3:])
        this_z=gsu.redshift_from_snapshot(snap_int)
        
        Npix=((1.0 + this_z)**(-1.0))*ckpcz/pixsize_kpc
        
        delt=np.int64(Npix/2)
        
    else:
        print('Size definiton required!')
        return axi
        
    r = r[dx+mid-delt:dx+mid+delt,dy+mid-delt:dy+mid+delt]
    g = g[dx+mid-delt:dx+mid+delt,dy+mid-delt:dy+mid+delt]
    b = b[dx+mid-delt:dx+mid+delt,dy+mid-delt:dy+mid+delt]

    axi.set_xticks([]) ; axi.set_yticks([])
            
    alph=alph ; Q=Q
            
    rgbthing = make_color_image.make_interactive(b,g,r,alph,Q)
    axi.imshow(rgbthing,interpolation='nearest',aspect='auto',origin='lower')

    if rfkey is not None:
        rf_fn = np.asarray(glob.glob(image_base+'/subdir_???/images_subhalo_'+str(subfindID)+'/*'+'cam'+camstr+'_'+rfkey+'_'+sb+'.fits' ))[0]
        try:
            axi = icmp.overplot_morph_data(axi,rf_fn,mid,delt,lw=2)
        except:
            pass
        
    return axi

