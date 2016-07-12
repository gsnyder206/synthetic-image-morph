import os
import shutil
import glob
import numpy as np
import astropy.io.ascii as ascii
import time


if __name__=="__main__":
    out_files = np.sort(np.asarray(glob.glob("group_15??_pbs.out")))
    total_wall_time_mins=0.0
    total_cpu_time_mins=0.0
    
    for of in out_files:
        with open(of,'r') as F:
            for line in F:
                if line.find('Resources Used') is not -1:
                    cpus = line[21:29]
                    walls = line[-8:]
                    print of, cpus, walls
                    
                    cputs = time.strptime(cpus,"%H:%M:%S")
                    wallts = time.strptime(walls,"%H:%M:%S")
                    cpu_mins = cputs.tm_hour*60.0 + cputs.tm_min + cputs.tm_sec/60.0
                    wall_mins = wallts.tm_hour*60.0 + wallts.tm_min + wallts.tm_sec/60.0
                    SUs = 16.0*wall_mins/60.0
                    eff = cpu_mins/(wall_mins*16.0)
                    print of, SUs, eff, cpu_mins, wall_mins
                    
