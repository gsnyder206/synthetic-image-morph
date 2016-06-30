
import os
import shutil
import glob
import numpy as np



def prep_mock_image(groupfile=None,output_dir='/home/gsnyder/oasis_project/PanSTARRS/mockimage_tests/',stubfolder='/home/gsnyder/sunrise_data/panstarrs_stubs/test',bindir='/home/gsnyder/bin', filters='/home/gsnyder/sunrise_data/sunrise_filters/lsst/'):


    base_name = os.path.basename(groupfile).rstrip('.hdf5')
    image_folder = os.path.join(output_dir,base_name+'_images')
    print image_folder

    if not os.path.lexists(image_folder):
        os.mkdir(image_folder)


    snapfile=groupfile
    exists = os.path.exists(snapfile)
    if exists==False:
        print snapfile+' does not exist, skipping...'
        return

    cwd = os.getcwd()
    os.chdir(image_folder)

    filter_dir = 'filters'
    if not os.path.lexists(filter_dir):
        os.mkdir(filter_dir)

    filter_file_list = np.asarray(glob.glob(os.path.join(filters,"*")))
    filter_text = open('filters_list','w')
    for ff in filter_file_list:
        fname = os.path.basename(ff)
        shutil.copy(ff,filter_dir)
        filter_text.write(fname+'\n')
    filter_text.close()

    stub_files = np.asarray(glob.glob(os.path.join(stubfolder,"*")))
    for sf in stub_files:
        shutil.copy(sf,'.')


    n='16'      #how many total processors do you want?
    tile='16'   #how many processors do you want to reserve per node?#location="~gsnyder/sunrise/bin_Apr13_2012"
    redshift = 0.05

    location=bindir  #location of sunrise binary executable files
    queue="normal"    #which queue to submit to? try short_parallel for debugging?  Maybe?
    snapexten=".hdf5"      #hack to get the script to recognize different names of simulation input files ".hdf5" for that extension, blank ("") for binary files


    bsubfile="sunrise_"+base_name+'.qsub'
    sfile='sfrhist_'+base_name+'.config'
    mfile='mcrx_'+base_name+'.config'
    bfile='broadband_'+base_name+'.config'

    bsubf = open(bsubfile,'w')
    bsubf.write('#!/bin/bash\n')
    bsubf.write('#PBS -q '+queue+'\n')
    bsubf.write('#PBS -N '+base_name+'\n')
    bsubf.write('#PBS -l nodes=1:ppn='+n+':native\n')
    bsubf.write('#PBS -l walltime=01:00:00\n')
    bsubf.write('#PBS -o '+base_name+'_pbs.out\n')
    bsubf.write('#PBS -e '+base_name+'_pbs.err\n')
    bsubf.write('#PBS -A hsc100\n')
    bsubf.write('#PBS -m abe \n')
    bsubf.write('#PBS -M gsnyder@stsci.edu \n')
    bsubf.write('#PBS -V \n\n')

    bsubf.write('export LD_LIBRARY_PATH=/home/gsnyder/opt:$LD_LIBRARY_PATH #GCNs do not access /usr/lib64/libtbb.so.2 for some reason \n')
    bsubf.write('cd '+image_folder+'\n')

    bsubf.write(location+'/sfrhist '+sfile+' 1> sfrhist_'+base_name+'.out 2>&1\n')
    bsubf.write(location+'/mcrx '+mfile+' 1> mcrx_'+base_name+'.out 2>&1\n')
    bsubf.write('ln -s /scratch/$USER/$PBS_JOBID/mcrx_'+base_name+'.fits .\n')
    bsubf.write(location+'/broadband '+bfile+' 1> broadband_'+base_name+'.out 2>&1\n')
    bsubf.write('rm -f /scratch/$USER/$PBS_JOBID/mcrx_'+base_name+'.fits\n')
    bsubf.write('rm -f /scratch/$USER/$PBS_JOBID/sfrhist_'+base_name+'.fits\n')
    bsubf.write('rm -f mcrx_'+base_name+'.fits\n')
    #bsubf.write('mv /scratch/$USER/$PBS_JOBID/broadband_'+base_name+'.fits .\n')
    bsubf.close()

    sf = open(sfile,'w')
    sf.write('#Lightcone Cylinder Parameter File for Sunrise, sfrhist\n')
    sf.write('include_file\t\t sfrhist_base.stub\n')
    sf.write('snapshot_file\t\t '+snapfile+'\n')
    sf.write('output_file\t\t /scratch/$USER/$PBS_JOBID/sfrhist_'+base_name+'.fits\n')
    #sf.write('translate_origin\t {:15.3f} {:15.3f} {:15.3f}\n'.format(float(v_In_phys_x[index]),float(v_In_phys_y[index]),float(v_In_phys_z[index])))
    #sf.write('grid_min		{:6.1f} {:6.1f} {:6.1f}        / [kpc]\n '.format( float(-5000.0/(1+z)),float(-5000.0/(1+z)),float(-5000.0/(1+z))   ))
    #sf.write('grid_max		{:7.1f} {:7.1f} {:7.1f}        / [kpc]\n '.format( float(110000.0/(1+z)),float(110000.0/(1+z)),float(110000.0/(1+z))   )+'\n')

                
    sf.close()

    mf = open(mfile,'w')
    mf.write('#Lightcone Cylinder Parameter File for Sunrise, mcrx\n')
    mf.write('include_file\t\t mcrx_base.stub\n')
    mf.write('input_file\t\t /scratch/$USER/$PBS_JOBID/sfrhist_'+base_name+'.fits\n')
    mf.write('output_file\t\t /scratch/$USER/$PBS_JOBID/mcrx_'+base_name+'.fits\n')
    #mf.write('camera_position\t\t {:15.3f} {:15.3f} {:15.3f}\n'.format(float(cam_phys_offset_x[index]),float(cam_phys_offset_y[index]),float(cam_phys_offset_z[index])))
    #mf.write('camera_direction\t {:15.8f} {:15.8f} {:15.8f}\n'.format(u3vec[0],u3vec[1],u3vec[2]))
    #mf.write('camera_up\t\t {:15.8f} {:15.8f} {:15.8f}\n'.format(u2vec[0],u2vec[1],u2vec[2]))
    #mf.write('camerafov\t\t {:15.10f}\n'.format(float(fov_rad[index])))
    #mf.write('translate_origin\t {:15.3f} {:15.3f} {:15.3f}\n'.format(float(v_In_phys_x[index]),float(v_In_phys_y[index]),float(v_In_phys_z[index]))+'\n')
    mf.close()

    bf = open(bfile,'w')
    bf.write('#Lightcone Cylinder Parameter File for Sunrise, broadband\n')
    bf.write('include_file\t\t broadband_base.stub\n')
    bf.write('input_file\t\t /scratch/$USER/$PBS_JOBID/mcrx_'+base_name+'.fits\n')
    bf.write('output_file\t\t broadband_'+base_name+'.fits\n')
    bf.write('redshift\t\t {:10.6f}\n'.format(float(redshift)))
    bf.close()

    return



if __name__=="__main__":
    prep_mock_image(groupfile='/home/gsnyder/oasis_project/PanSTARRS/GroupParsedSnapshots/snapshot_135/subfolder_001/group_150.hdf5')
