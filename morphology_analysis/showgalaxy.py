import astropy.io.fits as pyfits
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import scipy.ndimage
import scipy as sp

import make_color_image



def showgalaxy(subfindID,camera,filters=['NC-F115W','NC-F150W','NC-F200W'],alph=0.2,Q=8.0,Npix=400,sb='SB25',size=2):
    web_image_base='http://www.stsci.edu/~gsnyder/DtU2017/snapshot_068/subdir_010/images_subhalo_'
    webdir = web_image_base+str(subfindID)
    camstr = camera 
    #http://www.stsci.edu/~gsnyder/DtU2017/snapshot_068/subdir_010/images_subhalo_101278/snap068dir010sh101278cam00_ACS-F606W_SB27.fits
    b_fn = webdir+'/snap068dir010sh'+str(subfindID)+'cam'+camstr+'_'+filters[0]+'_'+sb+'.fits'
    g_fn = webdir+'/snap068dir010sh'+str(subfindID)+'cam'+camstr+'_'+filters[1]+'_'+sb+'.fits'
    r_fn = webdir+'/snap068dir010sh'+str(subfindID)+'cam'+camstr+'_'+filters[2]+'_'+sb+'.fits'
    #print('Accessing...',b_fn,g_fn,r_fn)

    try:
        r = pyfits.open(r_fn)[0].data
        g = pyfits.open(g_fn)[0].data
        b = pyfits.open(b_fn)[0].data
    except:
        print('Could not open files requested.')
        return -1
        
    mid = np.int64(r.shape[0]/2)
    delt=np.int64(Npix/2)
                
    r = r[mid-delt:mid+delt,mid-delt:mid+delt]
    g = g[mid-delt:mid+delt,mid-delt:mid+delt]
    b = b[mid-delt:mid+delt,mid-delt:mid+delt]


    fig=pyplot.figure(figsize=(size,size)) 
    axi = fig.add_subplot(1,1,1)
    axi.set_xticks([]) ; axi.set_yticks([])
            
    alph=alph ; Q=Q
            
    rgbthing = make_color_image.make_interactive(b,g,r,alph,Q)
    axi.imshow(rgbthing,interpolation='nearest',aspect='auto',origin='lower')
    pyplot.show()
    
    pyplot.close(fig)
    
    return 0



def listavailable():
    snap068_dir10_subid=[100589,
                         101193,
                         101510,
                         101978,
                         102831,
                         103129,
                         103955,
                         104136,
                         105100,
                         105230,
                         106124,
                         106198,
                         99794,
                         99982,
                         100072,
                         100126,
                         100199,
                         100282,
                         100381,
                         100426,
                         100519,
                         100760,
                         100832,
                         100950,
                         101029,
                         101102,
                         101278,
                         101322,
                         101387,
                         101459,
                         101681,
                         101764,
                         101836,
                         101903,
                         102052,
                         102098,
                         102158,
                         102242,
                         102423,
                         102424,
                         102478,
                         102591,
                         102671,
                         102672,
                         102883,
                         102955,
                         103050,
                         103185,
                         103254,
                         103298,
                         103376,
                         103438,
                         103527,
                         103590,
                         103701,
                         103771,
                         103834,
                         103899,
                         104010,
                         104081,
                         104185,
                         104290,
                         104354,
                         104421,
                         104469,
                         104540,
                         104629,
                         104725,
                         104797,
                         104884,
                         104946,
                         105017,
                         105151,
                         105325,
                         105403,
                         105478,
                         105552,
                         105608,
                         105693,
                         105746,
                         105747,
                         105789,
                         105827,
                         105897,
                         105980,
                         106051,
                         106247,
                         106326,
                         106388,
                         106472,
                         106546,
                         106627,
                         99450,
                         99451,
                         99533,
                         99624,
                         99701,
                         99793,
                         99842,
                         99916]
    
    return snap068_dir10_subid
