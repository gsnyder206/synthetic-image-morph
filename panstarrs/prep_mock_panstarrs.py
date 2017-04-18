#https://github.com/gsnyder206/mock-surveys
import illustris_api_utils as iau
import illustris_sunrise_utils as isu


def prep_mock_panstarrs():
    sim='Illustris-1'
    savepath=os.path.expandvars('$HOME/oasis_project_hsc102/IllustrisData/')
    use_z = 0.05
    
    #select snapshots and subhalos using web API locally or catalogs at STScI or harvard


    #loop over selected objects:

    for galaxy,sn,this_sfid in (selection object):
        
        f,s,d=iau.get_subhalo(sim,sn,this_sfid,savepath=savepath,verbose=True,clobber=False)

        #may want to create new function isu.setup_sunrise_illustris_panstarrs(f,s,redshift_override=use_z,filters='$MOCK_SURVEYS/tng/filters_lsst_light.txt')  ?
        isu.setup_sunrise_illustris_subhalo(f,s,redshift_override=use_z)
        #this also needs to be edited to include the candelization step in the job scripts, and output job submission scripts a la the lightcone function in "isu" module.
    



    return
