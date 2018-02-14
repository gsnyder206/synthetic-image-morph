#https://github.com/gsnyder206/mock-surveys
#~ import illustris_api_utils as iau
#~ import illustris_sunrise_utils as isu
#~ import gfs_sublink_utils as gsu
#~ import asciitable as ascii
#~ import glob

import numpy as np
import h5py
import os

import illustris_python as il


# ---------------------- GLOBAL VARIABLES ------------------------

parttype_list = [0, 1, 4, 5]  # gas, DM, stars, BHs
parttype_stars = 4

# ------------------------- FUNCTIONS ----------------------------

def _populate_group(group, npart_thisfile, basedir, snapnum, subfind_id, parttype):
    sub = il.snapshot.loadSubhalo(basedir, snapnum, subfind_id, parttype)
    for key in sub.keys():
        if key == 'count':
            npart_thisfile[parttype] = sub[key]
        elif key == 'ParticleIDs':
            # Careful: SUNRISE doesn't like big IDs, so we make up new ones...
            new_ids = np.arange(sub['count'], dtype=np.uint64)
            group.create_dataset(key, data=new_ids)
        else:
            group.create_dataset(key, data=sub[key])

def get_subhalo(basedir, writedir, snapnum, subfind_id):
    """Load all particles/cells for a given subhalo and store them
    in an HDF5 file that is readable by SUNRISE.
    
    Parameters
    ----------
    basedir : str
        Directory with the simulation output.
    writedir : str
        Base directory to write the HDF5 subhalo cutout.
    snapnum : int
        Snapshot number.
    subfind_id : array-like
        Subfind ID.
    
    Returns
    -------
    sub_filename : str
        Path to an HDF5 file containing the subhalo data.
    
    Notes
    -----
    - I should eventually implement a get_parent parameter, which loads
      the entire parent FoF group while pointing to the subhalo of interest.
    """

    # Create directory to store subhalo data
    sub_dir = '%s/snapnum_%03d/sub_%d' % (writedir, snapnum, subfind_id)
    if not os.path.lexists(sub_dir):
        os.makedirs(sub_dir)
    
    # Save subhalo data in HDF5 file
    sub_filename = '%s/cutout.hdf5' % (sub_dir)
    with h5py.File(sub_filename, 'w') as f_sub:
        # Create header for subhalo file
        sub_header = f_sub.create_group('Header')
        
        # We only need a handful of attributes, so we copy them by hand:
        keys = ['Time', 'HubbleParam', 'Omega0', 'OmegaLambda', 'MassTable', 'Redshift']
        snap_filename = '%s/snapdir_%03d/snap_%03d.0.hdf5' % (basedir, snapnum, snapnum)
        with h5py.File(snap_filename, 'r') as f_snap:
            snap_header = f_snap['Header']
            for key in keys:
                sub_header.attrs[key] = snap_header.attrs[key]
        
        # This attribute will be modified and added later:
        npart_thisfile = np.zeros(len(sub_header.attrs['MassTable']), dtype=np.int64)
        
        # Copy particle data from snapshot file to subhalo file,
        # one particle type at a time.
        for parttype in parttype_list:
            group = f_sub.create_group('PartType%d' % (parttype))
            _populate_group(group, npart_thisfile, basedir, snapnum, subfind_id, parttype)

        sub_header.attrs['NumPart_ThisFile'] = npart_thisfile

    return sub_filename

def generate_sfrhist_config(rundir, datadir, stubdir, sub_filename,
                            galprops, run_type, nthreads, scale_convert, num_rhalfs):
    """
    Based on a similar function from "illustris_sunrise_utils.py"
    """
    with open(rundir + '/sfrhist.config', 'w') as sf:
        sf.write('# Parameter file for SUNRISE, sfrhist\n\n')
        sf.write('include_file              %s\n' % (stubdir + '/sfrhist_base.stub'))
        sf.write('simparfile                %s\n' % (stubdir + '/simpar'))
        sf.write('snapshot_file             %s\n'% (sub_filename))
        sf.write('output_file               %s\n\n' % (rundir + '/sfrhist.fits'))
        sf.write('n_threads                 %d\n' % (nthreads))

        gridw = int(np.ceil(num_rhalfs*galprops['rhalf']*scale_convert))
        sf.write('translate_origin          %.2f\t%.2f\t%.2f         / [kpc]\n' % (
            galprops['pos_x']*scale_convert,
            galprops['pos_y']*scale_convert,
            galprops['pos_z']*scale_convert))
        sf.write('grid_min                  %.1f\t%.1f\t%.1f         / [kpc]\n' % (
            -1.0*gridw, -1.0*gridw,-1.0*gridw))
        sf.write('grid_max                  %.1f\t%.1f\t%.1f         / [kpc]\n\n' % (
            1.0*gridw, 1.0*gridw, 1.0*gridw))

        if run_type == 'images':
            sf.write('min_wavelength            %s\n' % ("0.02e-6"))
            sf.write('max_wavelength            %s\n\n' % ("5.0e-6"))
            sf.write('mappings_sed_file         %s\n' % (
                datadir + "/Smodel-lores128.fits"))
            sf.write('stellarmodelfile          %s\n' % (
                datadir + "/Patrik-imfKroupa-Zmulti-ml.fits"))

        elif run_type == 'ifu':
            sf.write('min_wavelength            %s\n' % ("0.6450e-6"))
            sf.write('max_wavelength            %s\n\n' % ("0.6650e-6"))
            sf.write('mappings_sed_file         %s\n' % (
                datadir + "/Smodel_full_hires.fits"))
            sf.write('stellarmodelfile          %s\n' % (
                datadir + "/logspace-Patrik-imfKroupa-geneva-Zmulti-hires.fits"))

        elif run_type == 'grism':
            sf.write('min_wavelength            %s\n' % ("0.02e-6"))
            sf.write('max_wavelength            %s\n\n' % ("5.0e-6"))
            sf.write('mappings_sed_file         %s\n' % (
                datadir+'/Mappings_Smodels_gfs.fits'))
            sf.write('stellarmodelfile          %s\n' % (
                datadir+'/GFS_combined_nolines.fits'))   

def generate_mcrx_config(rundir, stubdir, galprops, run_type, nthreads,
                         scale_convert, num_rhalfs, cam_file=None):
    """
    Based on a similar function from "illustris_sunrise_utils.py"
    """
    with open(rundir + '/mcrx.config', 'w') as mf:
        mf.write('# Parameter file for SUNRISE, mcrx\n\n')
        mf.write('include_file              %s\n' % (stubdir + '/mcrx_base.stub'))
        mf.write('input_file                %s\n' % (rundir + '/sfrhist.fits'))
        mf.write('output_file               %s\n' % (rundir + '/mcrx.fits'))

        #approximating Torrey and HST13887 settings
        if cam_file is None:
            mf.write('exclude_south_pole        true\n')
            mf.write('camerafov                 120\n')
            mf.write('ntheta                    2\n')
            mf.write('nphi                      3\n')
        else:
            mf.write('camera_positions          %s\n' % (cam_file))

        mf.write('n_threads                 %d\n' % (nthreads))
            
        if run_type != 'ifu':
            mf.write('use_kinematics            %s\n' % ('false  # true for IFU'))
        else:
            mf.write('use_kinematics            %s\n' % ('true  # false for images'))

        #move npixels to .config file
        npixels = 2*int(np.ceil(num_rhalfs*galprops['rhalf']*scale_convert))
        mf.write('npixels                   %d\n' % (npixels))

        mf.write('aux_particles_only        false\n')
        mf.write('nrays_nonscatter          %s' % ('1e7'))

def generate_broadband_config_images(rundir, datadir, stubdir, redshift):
    """
    Based on a similar function from "illustris_sunrise_utils.py"
    """
    with open(rundir + '/broadband.config','w') as bf:
        bf.write('# Parameter file for SUNRISE, broadband\n\n')
        bf.write('include_file                      %s\n\n' % (stubdir + '/broadband_base.stub'))
        bf.write('redshift                          %.1f\n\n' % (0.0))
        bf.write('input_file                        %s\n' % (rundir + '/mcrx.fits'))
        bf.write('output_file                       %s\n' % (rundir + '/broadband.fits'))
        bf.write('filter_list                       %s\n' % (datadir + 'sunrise_filters/filters_panstarrs'))
        bf.write('filter_file_directory             %s\n' % (datadir + 'sunrise_filters'))
    
    with open(rundir + '/broadbandz.config', 'w') as bfz:
        bfz.write('# Parameter file for SUNRISE, broadbandz\n\n')
        bfz.write('include_file                      %s\n\n' % (stubdir + '/broadband_base.stub'))
        bfz.write('redshift                          %.8f\n\n' % (redshift))
        bfz.write('input_file                        %s\n' % (rundir + '/mcrx.fits'))
        bfz.write('output_file                       %s\n' % (rundir + '/broadbandz.fits'))
        bfz.write('filter_list                       %s\n' % (datadir + 'sunrise_filters/filters_panstarrs'))
        bfz.write('filter_file_directory             %s\n' % (datadir + 'sunrise_filters'))

def generate_sbatch(rundir, nthreads):
    filepath = rundir + '/sunrise.sbatch'
    
    with open(filepath, 'w') as bsubf:
        bsubf.write('#!/bin/bash\n')
        bsubf.write('\n')
        bsubf.write('#SBATCH --mail-user=%s\n' % ('vrg@jhu.edu'))
        #bsubf.write('#SBATCH --mail-type=ALL\n')
        bsubf.write('#SBATCH -J sunrise\n')
        bsubf.write('#SBATCH -o sunrise.out\n')
        bsubf.write('#SBATCH -e sunrise.err\n')
        bsubf.write('#SBATCH -p hernquist\n')
        bsubf.write('#SBATCH -N 1\n')
        bsubf.write('#SBATCH --ntasks-per-node=%d\n' % (nthreads))
        bsubf.write('#SBATCH -t %s\n' % ('1-00:00:00'))
        bsubf.write('#SBATCH --export=ALL\n')
        bsubf.write('\n')

        bsubf.write('SYNIMAGE_CODE=$HOME/Python/PythonModules/synthetic-image-morph\n')
        bsubf.write('SUNRISE_BIN=$HOME/sunrise_env/bin\n')
        bsubf.write('\n')
        
        bsubf.write('cd ' + rundir + '\n')   # go to directory where job should run
        bsubf.write('${SUNRISE_BIN}/sfrhist sfrhist.config 1> sfrhist.out 2> sfrhist.err\n')
        bsubf.write('${SUNRISE_BIN}/mcrx mcrx.config 1> mcrx.out 2> mcrx.err\n')
        bsubf.write('${SUNRISE_BIN}/broadband broadbandz.config 1> broadbandz.out 2> broadbandz.err\n')
        bsubf.write('${SUNRISE_BIN}/broadband broadband.config 2> broadband.out 2> broadband.err\n')
        bsubf.write('\n')

    return os.path.abspath(filepath)

def setup_sunrise_subhalo(sub_filename, galprops, stubdir, datadir, num_rhalfs, use_z=None):
    """
    Based on a similar function from "illustris_sunrise_utils.py"
    """
    sub_dir = os.path.dirname(sub_filename)
    print("Using stubs in %s..." % (stubdir))
    
    list_of_types = ['images']
    nthreads = 24

    # Redshift can be obtained from subhalo cutout file
    with h5py.File(sub_filename, 'r') as f:
        real_z = f['Header'].attrs['Redshift']
        h = f['Header'].attrs['HubbleParam']

    scale_convert = 1.0 / (1.0 + real_z) / h  # to physical kpc
    
    if use_z is None:
        redshift = real_z
    else:
        redshift = use_z

    for run_type in list_of_types:
        rundir = '%s/%s' % (sub_dir, run_type)
        if not os.path.exists(rundir):
            os.makedirs(rundir)

        print('\tGenerating sfrhist.config file...')
        generate_sfrhist_config(
            rundir, datadir, stubdir, sub_filename, galprops,
            run_type, nthreads, scale_convert, num_rhalfs)

        print('\tGenerating mcrx.config file...')
        generate_mcrx_config(
            rundir, stubdir, galprops, run_type, nthreads, scale_convert,
            num_rhalfs, cam_file=None)

        print('\tGenerating broadband.config file...')
        generate_broadband_config_images(rundir, datadir, stubdir, redshift)

        print('\tGenerating sunrise.sbatch file...')
        batch_filename = generate_sbatch(rundir, nthreads)

        # ~ # SUNRISE expects "simpar" in the run directory
        # ~ os.popen('cp %s/simpar %s/' % (stubdir, rundir))
        
        # ~ # SUNRISE also asks for units.dat file
        # ~ os.popen('cp /n/home10/vrodrigu/sunrise_code/units.dat %s/' % (rundir))

    return batch_filename


def prep_mock_panstarrs(basedir, stubdir, writedir, datadir, snapnum, subfind_ids,
        num_rhalfs, use_z=0.05):
    """
    Parameters
    ----------
    basedir : str
        Directory with the simulation output.
    stubdir : str
        Directory with "stubs" for current simulation
    writedir : str
        Base directory to write the HDF5 subhalo cutout.
    datadir : str
        Directory with SUNRISE data
    snapnum : int
        Snapshot number
    subfind_ids : array-like
        List of Subfind IDs
    num_rhalfs : scalar
        How many stellar half-mass radii on each side from the center
    use_z : float, optional
        Assumed redshift
    """

    # Read some subhalo info
    sub_cm = il.groupcat.loadSubhalos(basedir, snapnum, fields=['SubhaloCM'])
    sub_pos = il.groupcat.loadSubhalos(basedir, snapnum, fields=['SubhaloPos'])
    sub_rhalf = il.groupcat.loadSubhalos(basedir, snapnum, fields=['SubhaloHalfmassRadType'])[:, parttype_stars]

    # Loop over selected objects
    for subfind_id in subfind_ids:
        # Get particle/cell data for current subhalo
        sub_filename = get_subhalo(basedir, writedir, snapnum, subfind_id)

        # Create "galprops" for this subhalo.
        galprops = {
            'pos_x':sub_pos[subfind_id, 0],
            'pos_y':sub_pos[subfind_id, 1],
            'pos_z':sub_pos[subfind_id, 2],
            'cm_x':sub_cm[subfind_id, 0],
            'cm_y':sub_cm[subfind_id, 1],
            'cm_z':sub_cm[subfind_id, 2],
            'rhalf':sub_rhalf[subfind_id],
        }
        
        # Create {sfrhist,mcrx,broadband}.config files
        batch_filename = setup_sunrise_subhalo(sub_filename, galprops, stubdir, datadir, num_rhalfs, use_z=use_z)

        #~ # This checks if subhalo exists, downloads it if not, and converts
        #~ # into SUNRISE-readable format.
        #~ # get_parent means it downloads the FOF group but points to each
        #~ # individual subhalo (duplicates data, but OK).
        #~ f,s,d = iau.get_subhalo(simulation, snapnums[i], subfind_ids[i],
                #~ savepath=savepath, verbose=True, clobber=False, get_parent=False)

        # may want to create new functions based around setup_sunrise_illustris_panstarrs(f,s,use_z=use_z,filters='$MOCK_SURVEYS/tng/filters_lsst_light.txt')  ?
        #examples in "isu" code:
        #isu.setup_sunrise_subhalo(f,s,use_z=use_z)
        
        # ~ script = setup_sunrise_subhalo(f,s,use_z=use_z)

        #this also needs to be edited to include the realism and morphology steps in the job scripts, and output job submission scripts a la the lightcone function in "isu" module.
        #script=setup_sunrise_illustris_panstarrs(f,s,use_z=use_z)        
        #the result will be all necessary snapshot data plus ancillary Sunrise data and input files, plus submission scripts

        # ~ #save "sbatch <script>" in text files for later use
        # ~ print(script)


    return
