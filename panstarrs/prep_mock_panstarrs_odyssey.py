#https://github.com/gsnyder206/mock-surveys
#~ import illustris_api_utils as iau
#~ import illustris_sunrise_utils as isu
#~ import gfs_sublink_utils as gsu
#~ import asciitable as ascii
#~ import glob

import numpy as np
import h5py
import os
import time

import illustris_python as il


# ---------------------- GLOBAL VARIABLES ------------------------

parttype_list = [0, 4, 5]  # gas, stars, BHs
parttype_gas = 0
parttype_stars = 4

# ------------------------- FUNCTIONS ----------------------------

def _populate_group(group, sub_header, npart_thisfile, basedir, snapnum,
                    subfind_id, parttype, nthreads):

    # Only load necessary fields
    if parttype == 0:
        # SofteningLength added later
        fields = ['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                  'StarFormationRate', 'InternalEnergy', 'Density',
                  'ElectronAbundance', 'GFM_Metallicity']
    elif parttype == 4:
        fields = ['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                  'GFM_Metallicity', 'GFM_StellarFormationTime']
    elif parttype == 5:
        fields = ['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                  'BH_Mass', 'BH_Mdot']
    else:
        raise NotImplementedError

    sub = il.snapshot.loadSubhalo(basedir, snapnum, subfind_id, parttype, fields=fields)
    for key in sub.keys():
        if key == 'count':
            npart_thisfile[parttype] = sub[key]
        elif key == 'ParticleIDs':
            # Careful: SUNRISE doesn't like big IDs, so we make up new ones...
            new_ids = np.arange(sub['count'], dtype=np.uint64)
            group.create_dataset(key, data=new_ids)
        elif key == 'GFM_Metals':
            # Metal fractions are wrong for both gas and stars...
            pass
        else:
            group.create_dataset(key, data=sub[key])

    if parttype == 0:
        # Instead of using the Arepo SmoothingLength from the snapshots,
        # we use twice the radius corresponding to the cell volume.
        if npart_thisfile[parttype] == 0:
            # Not much to do
            pass
        else:
            vol = sub['Masses'] / sub['Density']
            vol_radius = (3.0/(4.0*np.pi)*vol)**(1.0/3.0)
            smoothing_length = 2.0 * vol_radius
            group.create_dataset('SmoothingLength', data=smoothing_length)

def get_subhalo(basedir, writedir, snapnum, subfind_id, nthreads):
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
        keys = ['Time', 'HubbleParam', 'Omega0', 'OmegaLambda', 'MassTable', 'Redshift', 'BoxSize']
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
            _populate_group(group, sub_header, npart_thisfile, basedir, snapnum,
                            subfind_id, parttype, nthreads)

        sub_header.attrs['NumPart_ThisFile'] = npart_thisfile

        with_dust = True
        if npart_thisfile[parttype_gas] == 0:
            with_dust = False

    return sub_filename, with_dust

def generate_sfrhist_config(rundir, datadir, stubdir, sub_filename,
                            galprops, run_type, nthreads, nrays_per_pixel,
                            scale_convert, num_rhalfs, kpc_h_per_pixel):
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
        sf.write('work_chunk_levels         %d\n' % (2))

        half_npixels = int(np.ceil(num_rhalfs*galprops['rhalf']/kpc_h_per_pixel))
        # Approximately match Torrey/Snyder settings (grid = 4 * fov):
        gridw = 4 * half_npixels * kpc_h_per_pixel * scale_convert

        sf.write('translate_origin          %.2f\t%.2f\t%.2f         / [kpc]\n' % (
            galprops['pos_x']*scale_convert,
            galprops['pos_y']*scale_convert,
            galprops['pos_z']*scale_convert))
        sf.write('grid_min                  %.1f\t%.1f\t%.1f         / [kpc]\n' % (
            -1.0*gridw, -1.0*gridw,-1.0*gridw))
        sf.write('grid_max                  %.1f\t%.1f\t%.1f         / [kpc]\n\n' % (
            1.0*gridw, 1.0*gridw, 1.0*gridw))

        npixels = 2*half_npixels
        sf.write('n_rays_estimated          %d\n\n' % (nrays_per_pixel * npixels**2))

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

def generate_mcrx_config(rundir, stubdir, galprops, run_type, nthreads, nrays_per_pixel,
                         scale_convert, num_rhalfs, kpc_h_per_pixel, with_dust, cam_file=None):
    """
    Based on a similar function from "illustris_sunrise_utils.py"
    """
    with open(rundir + '/mcrx.config', 'w') as mf:
        mf.write('# Parameter file for SUNRISE, mcrx\n\n')
        mf.write('include_file              %s\n' % (stubdir + '/mcrx_base.stub'))
        mf.write('input_file                %s\n' % (rundir + '/sfrhist.fits'))
        mf.write('output_file               %s\n' % (rundir + '/mcrx.fits'))

        half_npixels = int(np.ceil(num_rhalfs*galprops['rhalf']/kpc_h_per_pixel))
        npixels = 2*half_npixels
        camerafov = npixels * kpc_h_per_pixel * scale_convert
        
        #approximating Torrey and HST13887 settings
        if cam_file is None:
            mf.write('exclude_south_pole        true\n')
            mf.write('camerafov                 %.1f\n' % (camerafov))
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
        mf.write('npixels                   %d\n' % (npixels))

        mf.write('aux_particles_only        false\n')
        mf.write('nrays_nonscatter          %d\n' % (nrays_per_pixel * npixels**2))
        if with_dust:
            mf.write('nrays_scatter             %d\n' % (nrays_per_pixel * npixels**2))
        else:
            mf.write('nrays_scatter             %d\n' % (0))
        mf.write('nrays_aux                 %d\n' % (nrays_per_pixel * npixels**2))


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
        bf.write('filter_list                       %s\n' % (datadir + '/sunrise_filters/filters_panstarrs'))
        bf.write('filter_file_directory             %s\n' % (datadir + '/sunrise_filters/'))  # need to include the trailing slash...
    
    with open(rundir + '/broadbandz.config', 'w') as bfz:
        bfz.write('# Parameter file for SUNRISE, broadbandz\n\n')
        bfz.write('include_file                      %s\n\n' % (stubdir + '/broadband_base.stub'))
        bfz.write('redshift                          %.8f\n\n' % (redshift))
        bfz.write('input_file                        %s\n' % (rundir + '/mcrx.fits'))
        bfz.write('output_file                       %s\n' % (rundir + '/broadbandz.fits'))
        bfz.write('filter_list                       %s\n' % (datadir + '/sunrise_filters/filters_panstarrs'))
        bfz.write('filter_file_directory             %s\n' % (datadir + '/sunrise_filters/'))  # need to include the trailing slash...

def generate_sbatch(rundir, nthreads):
    filepath = rundir + '/sunrise.sbatch'
    
    with open(filepath, 'w') as bsubf:
        bsubf.write('#!/bin/bash\n')
        bsubf.write('\n')
        bsubf.write('#SBATCH --mail-user=%s\n' % ('vrg@jhu.edu'))
        bsubf.write('#SBATCH --mail-type=ALL\n')
        bsubf.write('#SBATCH -J sunrise\n')
        bsubf.write('#SBATCH -o sunrise.out\n')
        bsubf.write('#SBATCH -e sunrise.err\n')
        bsubf.write('#SBATCH -p hernquist\n')
        bsubf.write('#SBATCH -N 1\n')
        bsubf.write('#SBATCH --ntasks-per-node=%d\n' % (nthreads))
        bsubf.write('#SBATCH --mem-per-cpu=1875\n')
        bsubf.write('#SBATCH -t %s\n' % ('2-00:00:00'))
        bsubf.write('#SBATCH --export=ALL\n')
        bsubf.write('\n')

        bsubf.write('SYNIMAGE_CODE=$HOME/Python/PythonModules/synthetic-image-morph\n')
        bsubf.write('SUNRISE_BIN=$HOME/sunrise_env/bin\n')
        bsubf.write('\n')
        
        bsubf.write('cd ' + rundir + '\n')   # go to directory where job should run
        bsubf.write('echo "Starting sfrhist stage..." 1>&2\n')
        bsubf.write('time ${SUNRISE_BIN}/sfrhist sfrhist.config 1> sfrhist.out 2> sfrhist.err\n')
        bsubf.write('echo "Starting mcrx stage..." 1>&2\n')
        bsubf.write('time ${SUNRISE_BIN}/mcrx mcrx.config 1> mcrx.out 2> mcrx.err\n')
        bsubf.write('echo "Starting broadbandz stage..." 1>&2\n')
        bsubf.write('time ${SUNRISE_BIN}/broadband broadbandz.config 1> broadbandz.out 2> broadbandz.err\n')
        # ~ bsubf.write('echo "Starting broadband stage..." 1>&2\n')
        # ~ bsubf.write('${SUNRISE_BIN}/broadband broadband.config 2> broadband.out 2> broadband.err\n')
        bsubf.write('\n')

    return os.path.abspath(filepath)

def setup_sunrise_subhalo(sub_filename, galprops, stubdir, datadir, num_rhalfs,
                          nthreads, kpc_h_per_pixel, nrays_per_pixel, with_dust, use_z=None):
    """
    Based on a similar function from "illustris_sunrise_utils.py"
    """
    sub_dir = os.path.dirname(sub_filename)
    print("Using stubs in %s..." % (stubdir))
    
    list_of_types = ['images']

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
            run_type, nthreads, nrays_per_pixel, scale_convert, num_rhalfs, kpc_h_per_pixel)

        print('\tGenerating mcrx.config file...')
        generate_mcrx_config(
            rundir, stubdir, galprops, run_type, nthreads, nrays_per_pixel,
            scale_convert, num_rhalfs, kpc_h_per_pixel, with_dust, cam_file=None)

        print('\tGenerating broadband.config file...')
        generate_broadband_config_images(rundir, datadir, stubdir, redshift)

        print('\tGenerating sunrise.sbatch file...')
        batch_filename = generate_sbatch(rundir, nthreads)

    return batch_filename


def prep_mock_panstarrs(basedir, stubdir, writedir, datadir, snapnum, subfind_ids,
        num_rhalfs, nthreads, kpc_h_per_pixel, nrays_per_pixel, use_z=None):
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
    start = time.time()
    print('Reading some info for all subhalos...')
    sub_cm = il.groupcat.loadSubhalos(basedir, snapnum, fields=['SubhaloCM'])
    sub_pos = il.groupcat.loadSubhalos(basedir, snapnum, fields=['SubhaloPos'])
    sub_rhalf = il.groupcat.loadSubhalos(basedir, snapnum, fields=['SubhaloHalfmassRadType'])[:, parttype_stars]
    print ('Time: %f s.' % (time.time() - start))

    # Loop over selected objects
    for subfind_id in subfind_ids:
        # Get particle/cell data for current subhalo
        start = time.time()
        print('Reading snapshot info...')
        sub_filename, with_dust = get_subhalo(basedir, writedir, snapnum, subfind_id, nthreads)
        print ('Time: %f s.' % (time.time() - start))

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
        batch_filename = setup_sunrise_subhalo(
            sub_filename, galprops, stubdir, datadir, num_rhalfs, nthreads,
            kpc_h_per_pixel, nrays_per_pixel, with_dust, use_z=use_z)

        print('Finished for subhalo %d.' % (subfind_id))

    return
