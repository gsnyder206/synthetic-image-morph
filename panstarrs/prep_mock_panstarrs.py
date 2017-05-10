#https://github.com/gsnyder206/mock-surveys
import illustris_api_utils as iau
import illustris_sunrise_utils as isu
import gfs_sublink_utils as gsu
import asciitable as ascii
### system modules
import os

#~ def setup_sunrise_illustris_panstarrs(f,s,redshift_override=0.05):

    #~ submitscript=''

    #~ return submitscript


def setup_sunrise_illustris_subhalo(snap_cutout,subhalo_object,verbose=True,clobber=True,
                                    stub_dir='$HOME/Python/PythonModules/mock-surveys/stubs_illustris/',
                                    data_dir='$HOME/sunrise_data/',
                                    nthreads=24,redshift_override=None,walltime_limit='02:00:00',use_scratch=True):

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


        
        if run_type == 'images': 
            print('\tGenerating broadband.config file for %s...'%run_type)
            broadband_fn   = 'broadband.config'
            broadband_stub = os.path.join(stub_dir,'broadband_base.stub')

            isu.generate_broadband_config_images(run_dir = run_dir, snap_dir = snap_dir, data_dir=data_dir, filename = broadband_fn, 
                                             stub_name = broadband_stub, 
                                             galprops_data = galprops_data, idx = idx,redshift=redshift,use_scratch=use_scratch)
        if run_type == 'grism': 
            print('\tGenerating broadband.config file for %s...'%run_type)
            broadband_fn   = 'broadband.config'
            broadband_stub = os.path.join(stub_dir, 'broadband_base.stub')

            isu.generate_broadband_config_grism(run_dir = run_dir, snap_dir = snap_dir, data_dir=data_dir, filename = broadband_fn, 
                                            stub_name = broadband_stub, 
                                            galprops_data = galprops_data, idx = idx,redshift=redshift,use_scratch=use_scratch)





        print('\tGenerating sunrise.sbatch file for %s...'%run_type)
        sbatch_fn   = 'sunrise.sbatch'		
        final_fn = isu.generate_sbatch(run_dir = run_dir, snap_dir = snap_dir, filename = sbatch_fn, 
                                 galprops_data = galprops_data, run_type = run_type,ncpus=nthreads,walltime=walltime_limit,use_scratch=use_scratch)

    
    return final_fn




def prep_mock_panstarrs(input_catalog):
    """
    Parameters
    ----------
    input_catalog : File name or newline-separated string
    
    """
    sim='Illustris-1'
    #savepath=os.path.expandvars('$IllustrisData')
    savepath='/oasis/projects/nsf/hsc102/vrg/IllustrisData/'
    use_z = 0.05
    
    #select snapshots and subhalos using web API locally or catalogs at STScI or harvard
    #e.g., based on input_catalog file, need snapshot number and SubfindID number

    #e.g.
    data=ascii.read(input_catalog)
    sn=data['snap']
    sfid=data['sfid']
    
    
    #loop over selected objects:
    for this_sn,this_sfid in zip(sn,sfid):

        #this checks if halo exists, downloads it if not, and converts into Sunrise-readable format
        f,s,d=iau.get_subhalo(sim,this_sn,this_sfid,savepath=savepath,verbose=True,clobber=False,getparent=False)
        #get_parent means it downloads the FOF group but points to each individual subhalo (duplicates data, but OK)

        #may want to create new functions based around setup_sunrise_illustris_panstarrs(f,s,redshift_override=use_z,filters='$MOCK_SURVEYS/tng/filters_lsst_light.txt')  ?
        #examples in "isu" code:
        #isu.setup_sunrise_illustris_subhalo(f,s,redshift_override=use_z)
        
        script = setup_sunrise_illustris_subhalo(snap_cutout,subhalo_object,redshift_override=use_z)

        #this also needs to be edited to include the realism and morphology steps in the job scripts, and output job submission scripts a la the lightcone function in "isu" module.
        #script=setup_sunrise_illustris_panstarrs(f,s,redshift_override=use_z)        
        #the result will be all necessary snapshot data plus ancillary Sunrise data and input files, plus submission scripts

        #save "sbatch <script>" in text files for later use
        print script


    return
