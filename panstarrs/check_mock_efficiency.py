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
    out_files = np.sort(np.asarray(glob.glob("/Users/gsnyder/Dropbox/Workspace/Computing/XSEDE/JULY2016/group_15??_pbs.out")))
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
    print "Mean SU: ", np.mean(np.asarray(SUs_list))
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
    eff_list = 100*np.asarray(eff_list)

    rand.seed(21)
    
    SUpergal_1 = np.median(SUs_list[rand.random_integers(0,N-1,1)])
    SUpergal_10 = np.median(SUs_list[rand.random_integers(0,N-1,10)])
    SUpergal_100 = np.median(SUs_list)
    effpergal_1 = np.median(eff_list[rand.random_integers(0,N-1,1)])
    effpergal_10 = np.median(eff_list[rand.random_integers(0,N-1,10)])
    effpergal_100 = np.median(eff_list)

    
    f1 = pyplot.figure(figsize=(5.0,4.0), dpi=150)
    pyplot.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)
    axi = f1.add_subplot(2,1,1)
    axi.locator_params(nbins=5,prune='both')

    axi.semilogx([1,10,100],[SUpergal_1,SUpergal_10,SUpergal_100],'ok',markersize=8)
    axi.errorbar([100],[SUpergal_100],yerr=msbs.MAD(np.asarray(SUs_list)),color='black')
    axi.set_xlim(0.5,2e5)
    axi.set_ylim(2.1,3.1)
    axi.annotate('trivial extrapolation\n (1 galaxy per job)',(6.0e3,SUpergal_100-0.5*msbs.MAD(np.asarray(SUs_list))),xycoords='data',ha='center',va='center',color='black',size=12)
    axi.arrow(120.0,SUpergal_100+msbs.MAD(np.asarray(SUs_list)),5.0e4,0.0,shape='full',width=0.007,color='black',length_includes_head=True,head_length=3.0e4,overhang=0.3)
    axi.set_xticklabels([])
    axi.set_ylabel('median SU/galaxy')
    axi.plot([170,170],[0.0,5.0],color='gray',linestyle='dashed',lw=3)
    axi.plot([1.05e5,1.05e5],[0.0,5.0],color='orange',linestyle='dotted',lw=3)
    axi.plot([1.05e5],[SUpergal_100+msbs.MAD(np.asarray(SUs_list))],'o',color='orange',markersize=8)
    
    axi.annotate("GCN tests",(10.0,2.9),xycoords='data',ha='center',va='center',color='gray',size=12)
    axi.annotate("Project I request\n 105,000 galaxies\n 277,200 SUs",(6.0e3,2.9),xycoords='data',ha='center',va='center',color='orange',size=12)

    axi = f1.add_subplot(2,1,2)
    axi.locator_params(nbins=5,prune='both')

    axi.semilogx([1,10,100],[effpergal_1,effpergal_10,effpergal_100],'ok',markersize=8)
    axi.errorbar([100],[effpergal_100],yerr=msbs.MAD(np.asarray(eff_list)),color='black')
    axi.set_xlim(0.5,2e5)
    axi.set_ylim(69.0,101.0)
    axi.set_xlabel('number of galaxies')
    axi.set_ylabel('median efficiency (%)',size=10)
    axi.plot([170,170],[0.0,500.0],color='gray',linestyle='dashed',lw=3)
    axi.plot([1.05e5,1.05e5],[0.0,5.0],color='orange',linestyle='dotted',lw=3)

    f1.savefig("sunrisescaling.pdf")
    pyplot.close(f1)


    
    f1 = pyplot.figure(figsize=(5.0,4.0), dpi=150)
    pyplot.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98,wspace=0.0,hspace=0.0)
    axi = f1.add_subplot(1,1,1)

    ngals_1 = np.asarray([1,10])
    cumtime_1 = 2.5*np.asarray([13.687,148.901])

    ngals_2 = np.asarray([10,100,300])
    cumtime_2 = 2.5*np.asarray([81.986,897.033,2654.3])

    ngals_4 = np.asarray([100,300,1000])
    cumtime_4 = 2.5*np.asarray([458.451,1338.938,4632.330])

    ngals_16 = np.asarray([100,300,1000])
    cumtime_16 = 2.5*np.asarray([205.54,393.668,1259.841])

    axi.loglog(ngals_1,1*cumtime_1/ngals_1,'*b',markersize=8)
    axi.loglog(ngals_2,2*cumtime_2/ngals_2,'sr',markersize=8)
    axi.loglog(ngals_4,4*cumtime_4/ngals_4,'^g',markersize=8)
    axi.loglog(ngals_16,16*cumtime_16/ngals_16,'ok',markersize=8)
    
    axi.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    axi.set_yticks([30,50,70,90])
    
    axi.set_xlim(5,2000.0)
    axi.set_ylim(30.0,100.0)
    axi.minorticks_off()

    axi.set_ylabel('used CPU-sec/galaxy')
    axi.set_xlabel('number of images')
    
    f1.savefig("morphscaling.pdf")
    pyplot.close(f1)

    
