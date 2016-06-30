
import os




def prep_mock_image(galfile=None,output_dir='/home/gsnyder/oasis_project/PanSTARRS/mockimage_tests/',stubfolder='/home/gsnyder/sunrise_data/panstarrs_stubs/test',bindir='/home/gsnyder/bin'):


    base_name = os.path.basename(galfile).rstrip('.hdf5')
    image_folder = os.path.join(output_dir,base_name+'_images')
    if not os.path.lexists(image_folder):
        os.mkdir(image_folder)





return



if __name__=="__main__":
    prep_mock_image(galfile='/home/gsnyder/oasis_project/PanSTARRS/GroupParsedSnapshots/snapshot_135/subfolder_001/group_150.hdf5')
