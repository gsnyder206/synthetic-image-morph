import pandas as pd
import astropy
import astropy.io.fits as fits
from astropy.table import Table
import os
import numpy as np
import illcan_multiplots as icmp

def select_data(i_all,j_all,h_all,zrange=None,mrange=None,snpix_limit=[1.0,3.0,3.0]):

    i_i=np.ones_like(i_all['CANDELS_ID'])==True
    j_i=np.ones_like(j_all['CANDELS_ID'])==True
    h_i=np.ones_like(h_all['CANDELS_ID'])==True
    
    if snpix_limit is not None:
        i_i=i_i*(i_all['SN_PIX_I'].values >= snpix_limit[0])
        j_i=j_i*(j_all['SN_PIX_J'].values >= snpix_limit[1])
        h_i=h_i*(h_all['SN_PIX_H'].values >= snpix_limit[2])

    if zrange is not None:
        i_i=i_i*(i_all['zbest'].values>=zrange[0])*(i_all['zbest'].values<zrange[1])
        j_i=j_i*(j_all['zbest'].values>=zrange[0])*(j_all['zbest'].values<zrange[1])
        h_i=h_i*(h_all['zbest'].values>=zrange[0])*(h_all['zbest'].values<zrange[1])
        
    if mrange is not None:
        i_i=i_i*(i_all['lmass'].values>=mrange[0])*(i_all['lmass'].values<mrange[1])
        j_i=j_i*(j_all['lmass'].values>=mrange[0])*(j_all['lmass'].values<mrange[1])
        h_i=h_i*(h_all['lmass'].values>=mrange[0])*(h_all['lmass'].values<mrange[1])        
        
    return i_all[i_i],j_all[j_i],h_all[h_i]
    

def merge_field(folder='/Users/gsnyder/Dropbox/Projects/PythonCode/candels',field='egs'):

    if field=='gds-n':
        folder=folder+'/gds-n'
    elif field=='gds-s':
        folder=folder+'/gds-s'
    else:
        folder=folder+'/'+field
        
        
    zbest_file=os.path.join(folder,field+'_zbest_multiband_v2.0.cat')
    zbest_fast=os.path.join(folder,field+'_zbest_multiband_v2.0.fout')
    zbest_sfrs=os.path.join(folder,field+'_zbest_sfrall_v2.0.cat')

    ifn=os.path.join(folder,field+'_acs_f814w_30mas_morph.fits')
    jfn=os.path.join(folder,field+'_wfc3_f125w_60mas_feb17_morph.fits')
    hfn=os.path.join(folder,field+'_wfc3_f160w_60mas_feb17_morph.fits')

    
    

    df_z=pd.read_csv(zbest_file,delim_whitespace=True,comment='#')
    df_f=pd.read_table(zbest_fast,delim_whitespace=True,comment='#')
    df_s=pd.read_csv(zbest_sfrs,delim_whitespace=True,comment='#')


    itab=Table(fits.open(ifn)[1].data)
    jtab=Table(fits.open(jfn)[1].data)
    htab=Table(fits.open(hfn)[1].data)

    i_df=itab.to_pandas()
    j_df=jtab.to_pandas()
    h_df=htab.to_pandas()
    #        datacols=['dGM20_x','fGM20_x','ASYM_x','D_x','CON_x','SNPIX_F814W','dGM20_y','fGM20_y','ASYM_y','D_y','CON_y','SNPIX_F160W']

    
    i_df['GMS_I']=icmp.SGM20(i_df['GINI_I'].values, i_df['M20_I'].values)
    j_df['GMS_J']=icmp.SGM20(j_df['GINI_J'].values, j_df['M20_J'].values)
    h_df['GMS_H']=icmp.SGM20(h_df['GINI_H'].values, h_df['M20_H'].values)

    i_df['GMF_I']=icmp.FGM20(i_df['GINI_I'].values, i_df['M20_I'].values)
    j_df['GMF_J']=icmp.FGM20(j_df['GINI_J'].values, j_df['M20_J'].values)
    h_df['GMF_H']=icmp.FGM20(h_df['GINI_H'].values, h_df['M20_H'].values)
    
    
    #merge mass into morph catalog
    df_i2 = pd.merge(df_f[['id','lmass']],i_df,left_on='id',right_on='CANDELS_ID')
    #merge z and sfr into morph catalog
    df_i = pd.merge(df_s[['CANDELS_ID','zbest','sfr_best']],df_i2,on='CANDELS_ID')

    #merge mass into morph catalog
    df_j2 = pd.merge(df_f[['id','lmass']],j_df,left_on='id',right_on='CANDELS_ID')
    #merge z and sfr into morph catalog
    df_j = pd.merge(df_s[['CANDELS_ID','zbest','sfr_best']],df_j2,on='CANDELS_ID')
    
    #merge mass into morph catalog
    df_h2 = pd.merge(df_f[['id','lmass']],h_df,left_on='id',right_on='CANDELS_ID')
    #merge z and sfr into morph catalog
    df_h = pd.merge(df_s[['CANDELS_ID','zbest','sfr_best']],df_h2,on='CANDELS_ID')

    
    
    return df_i,df_j,df_h






def load_all_candels():

    print('Loading/Merging CANDELS data')

    
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df3=pd.DataFrame()
    
    fields=['cos','egs','gds-n','gds-s','uds']
    for f in fields:
        df1_f,df2_f,df3_f=merge_field(field=f)
        df1=df1.append(df1_f)
        df2=df2.append(df2_f)
        df3=df3.append(df3_f)



    #D=0 should never happen?
    df1=df1.drop( df1[ df1['D_I']==0.0 ].index) 
    df2=df2.drop( df2[ df2['D_J']==0.0 ].index) 
    df3=df3.drop( df3[ df3['D_H']==0.0 ].index) 
    
    df1['logD_I']=np.log10(df1['D_I'].values)
    df2['logD_J']=np.log10(df2['D_J'].values)
    df3['logD_H']=np.log10(df3['D_H'].values)
    
    df1a=df1.dropna(axis='index',subset=['lmass','logD_I'])
    df2a=df2.dropna(axis='index',subset=['lmass','logD_J'])
    df3a=df3.dropna(axis='index',subset=['lmass','logD_H'])

    df1a = df1a.rename(columns={'C_I':'CON_I'})

    return df1a,df2a,df3a




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


