import os
import shutil
import glob
import numpy as np
import astropy.io.ascii as ascii


if __name__=="__main__":
    out_files = np.sort(np.asarray(glob.glob("group_15??_pbs.out")))
    total_wall_time_mins=0.0
    total_cpu_time_mins=0.0
    
    for of in out_files:
        with open(of,'r') as F:
            for line in F:
                if line.find('Resources Used') is not -1:
                    print line
                    
