import os
import shutil
import glob
import numpy as np
import astropy.io.ascii as ascii
import time
import medianstats_bootstrap as msbs
import matplotlib
#matplotlib.use('PDF')
import matplotlib.pyplot as pyplot
import matplotlib.colors as pycolors
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import numpy.random as rand



if __name__=="__main__":
    out_files = np.sort(np.asarray(glob.glob("group_15??_pbs.out")))
    total_wall_time_mins=0.0
    total_cpu_time_mins=0.0
    SUs_list = []
    eff_list = []
    cpu_mins_list = []
    wall_mins_list = []
    
    for of in out_files:
        with open(of,'r') as F:
            for line in F:
                if line.find('Resources Used') is not -1:
                    cpus = line[21:29]
                    walls = line[-9:-1]  #account for \n
                    
                    cputs = time.strptime(cpus,"%H:%M:%S")
                    wallts = time.strptime(walls,"%H:%M:%S")
                    cpu_mins = cputs.tm_hour*60.0 + cputs.tm_min + cputs.tm_sec/60.0
                    wall_mins = wallts.tm_hour*60.0 + wallts.tm_min + wallts.tm_sec/60.0
                    SUs = 16.0*wall_mins/60.0
                    eff = cpu_mins/(wall_mins*16.0)
                    total_wall_time_mins = total_wall_time_mins +wall_mins
                    total_cpu_time_mins = total_cpu_time_mins +cpu_mins
                    SUs_list.append(SUs)
                    eff_list.append(eff)
                    cpu_mins_list.append(cpu_mins)
                    wall_mins_list.append(wall_mins)
                    
                    print of, SUs, eff, cpu_mins, wall_mins
                    

    print "Mean efficiency: ", total_cpu_time_mins/(total_wall_time_mins*16.0)
    print "Median SU: ", np.median(np.asarray(SUs_list))
    print "MAD SU: ", msbs.MAD(np.asarray(SUs_list))
    print "Median eff:", np.median(np.asarray(eff_list))
    print "MAD eff: ", msbs.MAD(np.asarray(eff_list))
    print "Median wall mins:", np.median(np.asarray(wall_mins_list))
    print "MAD wall mins: ", msbs.MAD(np.asarray(wall_mins_list))
    print " "
    print "#su  eff  cpu_mins  wall_mins  "
    for i in range(len(SUs_list)):
        
        print "{:6.3f}  {:6.3f}  {:6.3f}  {:6.3f}  ".format( SUs_list[i], eff_list[i], cpu_mins_list[i], wall_mins_list[i])
        

    N = len(SUs_list)
    
    SUpergal_1 = np.sum(SUs_list[rand.random_integers(0,N-1,1)])/1.0
    SUpergal_10 = np.sum(SUs_list[rand.random_integers(0,N-1,10)])/1.0
    SUpergal_100 = np.sum(SUs_list[rand.random_integers(0,N-1,100)])/1.0

    f1 = pyplot.figure(figsize=(3.0,2.0), dpi=150)
    pyplot.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.98,wspace=0.0,hspace=0.0)
    axi = f1.add_subplot(3,1,1)
    
    axi.semilogx([1,10,100],[SUpergal_1,SUpergal_10,SUpergal_100],'ok')
    axi.set_xlim(0.5,1.2e5)
    axi.set_ylim(2.2,3.2)
    
    
    f1.savefig("/home/gsnyder/PythonCode/synthetic-image-morph/panstarrs/sunrisescaling.pdf")
    pyplot.close(f1)

    
