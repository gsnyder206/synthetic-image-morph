
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.colors as pycolors
import matplotlib.cm as cm

import numpy as np
import numpy.ma as ma


def get_levels_from_percentiles(hist,tiles):

    minval = np.min(hist)
    maxval = np.max(hist)
    thresh = np.linspace(maxval,minval,1000)
    levels = []
    tot = np.sum(hist)
    #print hist.shape

    for t in tiles:
        frac = 0.0
        for l in thresh:
            wi = np.where(hist > l)
            #print wi
            if len(wi) > 0:
                frac = np.sum((hist)[wi])/tot
                #print frac, l, t, tot, wi
                if frac > t:
                    levels.append(l)
                    break

    return levels




def median_grid(indices,other_dict,**kwargs):

    x = other_dict['x']
    
    val = np.median(x[indices])

    return val,1

def fraction_grid(indices,other_dict,**kwargs):

    #x must be a boolean array
    x = other_dict['x']
    denominator = x[indices].shape[0]
    xw = np.where(x[indices])[0]
    numerator = xw.shape[0]

    
    val = float(numerator)/float(denominator)
    
    return val,1


def log_fraction_grid(indices,other_dict,**kwargs):

    #x must be a boolean array
    x = other_dict['x']
    denominator = x[indices].shape[0]
    xw = np.where(x[indices])[0]
    numerator = xw.shape[0]

    #print(numerator, denominator)
    val = np.where(numerator > 0, np.log10( float(numerator)/float(denominator)),-99.0 )
    
    return val,1

def normed_proportion_grid(indices,other_dict,hist_2d=None,**kwargs):

    N = np.sum(np.ones_like(hist_2d))
    
    #x must be a boolean array
    x = other_dict['x']
    denominator = x.shape[0] #x[indices].shape[0]
    
    xw = np.where(x[indices])[0]
    numerator = xw.shape[0]

    
    val = N*float(numerator)/float(denominator)
    
    return val,val


def logged_proportion_grid(indices,other_dict,hist_2d=None,**kwargs):

    N = np.sum(np.ones_like(hist_2d))
    
    #x must be a boolean array
    x = other_dict['x']
    denominator = x.shape[0] #x[indices].shape[0]
    
    xw = np.where(x[indices])[0]
    numerator = xw.shape[0]

    
    #val = N*float(numerator)/float(denominator)
    val = np.where(numerator > 0, np.log10( N*float(numerator)/float(denominator)),-99.0 )
    
    return val,val




def summed_logssfr(indices,other_dict,**kwargs):

    sfr = np.sum( (other_dict['sfr'])[indices] )
    mass = np.sum( (other_dict['mstar'])[indices] )

    val = np.log10(sfr)-np.log10(mass)
    #print(sfr,mass,val)
    
    return val,1


def test_function():
    print('I am a test function')
    return 1.0,1


def execute_twodim(xparam,yparam,xlim,ylim,other_params,other_function,bins=20,min_bin=3,vmin=0.0,bad_offset=-1000.0):
    
    MR_hist,xMR,yMR = np.histogram2d(xparam,yparam,bins=bins, range=[xlim, ylim],normed=False)
    dx = xMR[1] - xMR[0]
    dy = yMR[1] - yMR[0]

    value_twod = np.zeros_like(MR_hist) + bad_offset
    norm2d = np.ones_like(MR_hist)

    i=0
    for xe in xMR[1:]:
        j=0
        for ye in yMR[1:]:
            fli = np.where( np.logical_and(np.logical_and(np.logical_and( xparam < xe, xparam >= xe-dx ), yparam < ye),yparam >=ye-dy ) )[0]
            number = 0.0
            denom = 1.0
            if (fli.shape)[0] >= min_bin:
                
                #result = eval(other_function,{'indices':fli,'other_dict':other_params})
                result,norm = eval(other_function+'(fli,other_params,hist_2d=MR_hist)')

                value_twod[i,j] = result
                norm2d[i,j]=norm
                
            j = j+1
        i = i+1

    value_twod = np.where(value_twod > bad_offset, value_twod/np.max(norm2d), np.ones_like(value_twod)*bad_offset)

    return value_twod, MR_hist, xMR, yMR, dx, dy



def make_twod_grid(axi,xparam,yparam,other_params,other_function,bins=20,tiles=(0.90,0.50),percentiles=False,numbers=[10,100],vmin=0.0,vmax=1.0,bad_offset=-1000.0,flipx=False,**bin_kwargs):


    xlim = axi.get_xlim()
    ylim = axi.get_ylim()

    if flipx is True:
        xlim_use = np.flipud(xlim)
    else:
        xlim_use = xlim
        
    value_twod, hist_twod, xMR, yMR, dx, dy = execute_twodim(xparam,yparam,xlim_use,ylim,other_params,other_function,bins=bins,vmin=vmin,**bin_kwargs) #passive_fractions_twodim(xparam,yparam,xlimits,ylimits,CONDITIONAL,sfr_all,mstar_all,bins=bins)

    
    themap = cm.viridis
    Zm = ma.masked_where(value_twod <= bad_offset , value_twod)
    
    
    cscale_function = pycolors.Normalize(vmin=vmin,vmax=vmax,clip=True)

    colorobj = axi.imshow(np.transpose((Zm)),vmin=vmin,vmax=vmax,origin='lower',aspect='auto',interpolation='nearest',extent=[xlim_use[0],xlim_use[1],ylim[0],ylim[1]],cmap=themap)
        
    if percentiles==True:
        levels = get_levels_from_percentiles(MR_hist*1.0,tiles)
        axi.contour(xMR[1:]-dx/2.0,yMR[1:]-dy/2.0,np.transpose(np.log10(MR_hist+0.001)),np.log10(levels),linewidths=1.0,colors=['silver','white'])
    else:
        axi.contour(xMR[1:]-dx/2.0,yMR[1:]-dy/2.0,np.transpose(np.log10(hist_twod+0.001)),(np.log10(numbers[0]),np.log10(numbers[1])),linewidths=0.5,colors=['silver','white'])




    return axi, colorobj




def make_colorbar(colorcontourobject, fig, loc=[0.15,0.95,0.75,0.049], title='value',ticks=[0.0,0.5,1.0],fontsize=10,labelpad=1,format='%.1f'):


    fake_subplot = fig.add_axes(loc, frameon=True)
    #fake_subplot.set_axis_off()
    fake_subplot.set_xticks([]) ; fake_subplot.set_yticks([])

    cbar = pyplot.colorbar(colorcontourobject,cax=fake_subplot, orientation='horizontal',ticks=ticks,drawedges=False,format=format,extend='both',extendfrac=0.10)
    cbar.ax.tick_params(axis='both', which='major', labelsize=fontsize)
    cbar.ax.set_xlabel(title,size=fontsize,labelpad=labelpad)
