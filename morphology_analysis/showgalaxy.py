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


def showgalaxy(axi,snapkey,subfindID,camera,filters=['NC-F115W','NC-F150W','NC-F200W'],alph=0.2,Q=8.0,Npix=400,sb='SB25',rfkey=None):
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
    except:
        print('Could not open files requested.')
        return axi
        
    mid = np.int64(r.shape[0]/2)
    delt=np.int64(Npix/2)
                
    r = r[mid-delt:mid+delt,mid-delt:mid+delt]
    g = g[mid-delt:mid+delt,mid-delt:mid+delt]
    b = b[mid-delt:mid+delt,mid-delt:mid+delt]

    axi.set_xticks([]) ; axi.set_yticks([])
            
    alph=alph ; Q=Q
            
    rgbthing = make_color_image.make_interactive(b,g,r,alph,Q)
    axi.imshow(rgbthing,interpolation='nearest',aspect='auto',origin='lower')

    if rfkey is not None:
        rf_fn = np.asarray(glob.glob(image_base+'/subdir_???/images_subhalo_'+str(subfindID)+'/*'+'cam'+camstr+'_'+rfkey+'_'+sb+'.fits' ))[0]
        try:
            axi = icmp.overplot_morph_data(axi,rf_fn,mid,delt,lw=0.5)
        except:
            pass
        
    return axi

