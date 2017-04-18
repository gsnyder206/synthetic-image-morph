import candelize


if __name__=="__main__":
    res = candelize.process_snapshot(subdirpath='.',clobber=False,seg_filter_label='NC-F200W',magsb_limits=[25.0,27.0],
                                     camindices=[0],do_idl=False,analyze=True,use_nonscatter=True,Np=10)
