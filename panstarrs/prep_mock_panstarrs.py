#https://github.com/gsnyder206/mock-surveys
import illustris_api_utils as iau
import illustris_sunrise_utils as isu
### system modules
import os


def prep_mock_panstarrs(input_catalog):
    sim='Illustris-1'
    savepath=os.path.expandvars('$IllustrisData')  #want it at /oasis/projects/nsf/hsc102/vrg/IllustrisData/ maybe?  
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
        f,s,d=iau.get_subhalo(sim,this_sn,this_sfid,savepath=savepath,verbose=True,clobber=False,get_parent=True)  

        #may want to create new functions based around setup_sunrise_illustris_panstarrs(f,s,redshift_override=use_z,filters='$MOCK_SURVEYS/tng/filters_lsst_light.txt')  ?
        isu.setup_sunrise_illustris_subhalo(f,s,redshift_override=use_z)
        #this also needs to be edited to include the candelization step in the job scripts, and output job submission scripts a la the lightcone function in "isu" module.

        #the result will be all necessary snapshot data plus ancillary Sunrise data and input files, plus submission scripts



    return
