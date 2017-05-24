#https://github.com/gsnyder206/mock-surveys
import illustris_api_utils as iau
import illustris_sunrise_utils as isu
import gfs_sublink_utils as gsu
import asciitable as ascii
### system modules
import numpy as np
import os
import glob

#~ def setup_sunrise_illustris_panstarrs(f,s,redshift_override=0.05):

    #~ submitscript=''

    #~ return submitscript

def generate_sbatch(run_dir, run_type='images', ncpus='24', queue='compute',
        email='vrg@jhu.edu', walltime='04:00:00', account='hsc102', use_scratch=False):
    filepath = run_dir + '/sunrise.sbatch'
    bsubf = open(filepath, 'w+')
    bsubf.write('#!/bin/bash\n')
    bsubf.write('\n')
    bsubf.write('#SBATCH --mail-user=%s\n' % (email))
    #bsubf.write('#SBATCH --mail-type=ALL\n')
    bsubf.write('#SBATCH -J sunrise_%s\n' % (run_type))
    bsubf.write('#SBATCH -o sunrise_%s.out\n' % (run_type))
    bsubf.write('#SBATCH -e sunrise_%s.err\n' % (run_type))
    bsubf.write('#SBATCH -A %s \n' % (account))
    bsubf.write('#SBATCH --partition=%s\n' % (queue))
    bsubf.write('#SBATCH --nodes=1\n')
    bsubf.write('#SBATCH --ntasks-per-node=%s\n' % (ncpus))
    bsubf.write('#SBATCH -t %s\n' % (walltime))
    bsubf.write('#SBATCH --export=ALL\n')
    bsubf.write('\n')

    bsubf.write('SYNIMAGE_CODE=$HOME/Python/PythonModules/synthetic-image-morph\n')
    bsubf.write('\n')
    
    bsubf.write('cd ' + run_dir + '\n')   # go to directory where job should run
    bsubf.write('/home/gsnyder/bin/sfrhist sfrhist.config 1> sfrhist.out 2> sfrhist.err\n')
    bsubf.write('/home/gsnyder/bin/mcrx mcrx.config 1> mcrx.out 2> mcrx.err\n')
    if run_type=='images':
        #for these, may want to use:  https://github.com/gsnyder206/synthetic-image-morph/blob/master/tng/filters_lsst_light.txt
        bsubf.write('/home/gsnyder/bin/broadband broadbandz.config 1> broadbandz.out 2> broadbandz.err\n')
        bsubf.write('/home/gsnyder/bin/broadband broadband.config 2> broadband.out 2> broadband.err\n')
        bsubf.write(os.path.expandvars('python $SYNIMAGE_CODE/panstarrs/mock_panstarrs.py\n'))
    elif run_type=='ifu':
        bsubf.write('rm -rf sfrhist.fits\n')   #enable this after testing
        #bsubf.write('gzip -9 mcrx.fits\n')
    elif run_type=='grism':
        bsubf.write('/home/gsnyder/bin/broadband broadbandgrism.config > broadbandgrism.out 2> broadbandgrism.err\n')
        #bsubf.write('rm -rf sfrhist.fits\n')   #enable this after testing
        #bsubf.write('rm -rf mcrx.fits\n')   #enable this after testing

    if use_scratch is True:
        bsubf.write('cp /scratch/$USER/$SLURM_JOBID/broadband*.fits .')
    
    bsubf.write('\n')
    bsubf.close()

    return os.path.abspath(filepath)


def generate_broadband_config_panstarrs(run_dir, snap_dir, data_dir, filename,
        stub_name, galprops_data, idx = None,redshift=0.0,use_scratch=False,isnap=None):

    #copy sunrise filter folder to snap_dir+'/inputs/sunrise_filters/'

    bf = open(run_dir+'/'+filename,'w+')
    red0=0.0
        

    if use_scratch is True:
        if isnap is not None:
            int_dir='/scratch/$USER/$SLURM_JOBID/'+str(isnap)
        else:
            int_dir='/scratch/$USER/$SLURM_JOBID'
    else:
        int_dir=run_dir


    bf.write('#Parameter File for Sunrise, broadband\n\n')
    bf.write('include_file                      %s\n\n'%stub_name)
    bf.write('redshift                          %.3f\n\n'%red0)
    bf.write('input_file                        %s\n'%(int_dir+'/mcrx.fits'))
    bf.write('output_file                       %s\n'%(int_dir+'/broadband.fits'))
    bf.write('filter_list                       %s\n'%('/home/vrg/Python/PythonModules/synthetic-image-morph/tng/filters_lsst_light.txt'))
    bf.write('filter_file_directory             %s\n'%(data_dir+'sunrise_filters/'))
    bf.close()
    
    bfz = open(run_dir+'/'+filename.replace('broadband','broadbandz'),'w+')
    
    bfz.write('#Parameter File for Sunrise, broadband\n\n')
    bfz.write('include_file                      %s\n\n'%stub_name)
    bfz.write('redshift                          %.8f\n\n'%redshift)
    bfz.write('input_file                        %s\n'%(int_dir+'/mcrx.fits'))
    bfz.write('output_file                       %s\n'%(int_dir+'/broadbandz.fits'))
    bfz.write('filter_list                       %s\n'%('/home/vrg/Python/PythonModules/synthetic-image-morph/tng/filters_lsst_light.txt'))
    bfz.write('filter_file_directory             %s\n'%(data_dir+'sunrise_filters/'))
    bfz.close()
    

    print('\t\tSuccessfully generated %s'%filename)

    return


def setup_sunrise_illustris_subhalo(snap_cutout, subhalo_object, verbose=True, clobber=True,
        stub_dir='$HOME/Python/PythonModules/mock-surveys/stubs_illustris/',
        data_dir='$HOME/sunrise_data/', nthreads=24, redshift_override=None,
        walltime_limit='02:00:00', use_scratch=False):

    fits_file = os.path.abspath(snap_cutout)
    galprops_data = subhalo_object
    
    snap_dir = os.path.dirname(fits_file)
    
    print("Setting up sunrise run in.. ", snap_dir)

    stub_dir = os.path.expandvars(stub_dir)
    data_dir = os.path.expandvars(data_dir)
    
    print("Using stubs in.. ",stub_dir)
    stub_files = np.asarray(glob.glob(os.path.join('stub_dir','*')))

    
    list_of_types = ['images']

    idx=None

    real_redshift=gsu.redshift_from_snapshot( subhalo_object['snap'] )
    scale_convert=(1.0/(gsu.ilh*(1.0 + real_redshift)))
    
    if redshift_override is None:
        redshift=real_redshift
    else:
        redshift=redshift_override


    print('redshift= ',redshift)
    
    nthreads=str(nthreads)
        
    for run_type in list_of_types:
        run_dir = snap_dir+'/%s'%run_type
        if not os.path.lexists(run_dir):
            os.mkdir(run_dir)

        for sf in stub_files:
            shutil.copy(sf,run_dir)
            
        print('\tGenerating sfrhist.config file for %s...'%run_type)
        sfrhist_fn   = 'sfrhist.config'
        sfrhist_stub = os.path.join(stub_dir,'sfrhist_base.stub')

        isu.generate_sfrhist_config(run_dir = run_dir, filename = sfrhist_fn, data_dir=data_dir,
                                stub_name = sfrhist_stub,  fits_file = fits_file, 
                                galprops_data = galprops_data, run_type = run_type,
                                nthreads=nthreads, idx = idx,scale_convert=scale_convert,use_scratch=use_scratch)

        print('\tGenerating mcrx.config file for %s...'%run_type)
        mcrx_fn   = 'mcrx.config'
        mcrx_stub = os.path.join(stub_dir,'mcrx_base.stub')

        isu.generate_mcrx_config(run_dir = run_dir, snap_dir = snap_dir, filename = mcrx_fn, 
                             stub_name = mcrx_stub,
                             galprops_data = galprops_data, run_type = run_type, nthreads=nthreads, cam_file=None , idx = idx,use_scratch=use_scratch)
        
        print('\tGenerating broadband.config file for %s...'%run_type)
        broadband_fn   = 'broadband.config'
        broadband_stub = os.path.join(stub_dir,'broadband_base.stub')

        generate_broadband_config_panstarrs(run_dir = run_dir, snap_dir = snap_dir, data_dir=data_dir, filename = broadband_fn, 
                                         stub_name = broadband_stub, 
                                         galprops_data = galprops_data, idx = idx,redshift=redshift,use_scratch=use_scratch)

        print('\tGenerating sunrise.sbatch file for %s...'%run_type)
        final_fn = generate_sbatch(run_dir, run_type=run_type, ncpus=nthreads, queue='compute',
                email='vrg@jhu.edu', walltime='04:00:00', account='hsc102', use_scratch=use_scratch)

        # SUNRISE asks for a ./simpar file, so we copy it from mock-surveys
        os.popen('cp %s/simpar %s/' % (stub_dir, run_dir))
        
        # SUNRISE also asks for units.dat file
        os.popen('cp /home/vrg/etc/units.dat %s/' % (run_dir))

    return final_fn


def prep_mock_panstarrs(snapnums, subfind_ids, simulation='Illustris-1', use_z=0.05,
        savepath='/oasis/projects/nsf/hsc102/vrg/IllustrisData'):
    """
    Parameters
    ----------
    snapnums : array-like
        List of snapshot numbers
    subfind_ids : array-like
        List of Subfind IDs
    simulation : str
        Name of the simulation
    use_z : float
        Assumed redshift
    savepath : str
        Where to store the images
    """
    assert(len(snapnums) == len(subfind_ids))
    
    sim='Illustris-1'
    use_z = 0.05
    
    # Loop over selected objects
    for i in xrange(len(snapnums)):
        # this checks if halo exists, downloads it if not, and converts
        # into Sunrise-readable format
        f,s,d = iau.get_subhalo(simulation, snapnums[i], subfind_ids[i],
                savepath=savepath, verbose=True, clobber=False, getparent=False)
        # getparent means it downloads the FOF group but points to each
        # individual subhalo (duplicates data, but OK)

        # may want to create new functions based around setup_sunrise_illustris_panstarrs(f,s,redshift_override=use_z,filters='$MOCK_SURVEYS/tng/filters_lsst_light.txt')  ?
        #examples in "isu" code:
        #isu.setup_sunrise_illustris_subhalo(f,s,redshift_override=use_z)
        
        script = setup_sunrise_illustris_subhalo(f,s,redshift_override=use_z)

        #this also needs to be edited to include the realism and morphology steps in the job scripts, and output job submission scripts a la the lightcone function in "isu" module.
        #script=setup_sunrise_illustris_panstarrs(f,s,redshift_override=use_z)        
        #the result will be all necessary snapshot data plus ancillary Sunrise data and input files, plus submission scripts

        #save "sbatch <script>" in text files for later use
        print script


    return
