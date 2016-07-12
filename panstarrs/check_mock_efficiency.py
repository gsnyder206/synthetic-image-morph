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

    SUs_list = np.asarray(SUs_list)
    
    SUpergal_1 = np.sum(SUs_list[rand.random_integers(0,N-1,1)])/1.0
    SUpergal_10 = np.sum(SUs_list[rand.random_integers(0,N-1,10)])/10.0
    SUpergal_100 = np.sum(SUs_list[rand.random_integers(0,N-1,100)])/100.0

    f1 = pyplot.figure(figsize=(5.0,4.0), dpi=150)
    pyplot.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98,wspace=0.0,hspace=0.0)
    axi = f1.add_subplot(2,1,1)
    
    axi.semilogx([1,10,100],[SUpergal_1,SUpergal_10,SUpergal_100],'ok')
    axi.set_xlim(0.5,1.2e5)
    axi.set_ylim(2.2,3.2)
    
    
    f1.savefig("sunrisescaling.pdf")
    pyplot.close(f1)


    
    f1 = pyplot.figure(figsize=(5.0,4.0), dpi=150)
    pyplot.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98,wspace=0.0,hspace=0.0)
    axi = f1.add_subplot(1,1,1)
    
    ngals_1 = np.asarray([1,10])
    cumtime_1 = np.asarray([13.687,148.901])

    ngals_2 = np.asarray([10,100])
    cumtime_2 = np.asarray([81.986,897.033])

    ngals_4 = np.asarray([100])
    cumtime_4 = np.asarray([458.451])

    ngals_16 = np.asarray([100,1000])
    cumtime_16 = np.asarray([187.22,1259.841])
    
    axi.semilogx(ngals_1,1*cumtime_1/ngals_1,'*b')
    axi.semilogx(ngals_2,2*cumtime_2/ngals_2,'sr')
    axi.semilogx(ngals_4,4*cumtime_4/ngals_4,'^g')
    axi.semilogx(ngals_16,16*cumtime_16/ngals_16,'ok',markersize=3)
    axi.set_xlim(0.5,1200.0)
    axi.set_ylim(10.0,60.0)
    
    f1.savefig("morphscaling.pdf")
    pyplot.close(f1)

    
