#!/usr/bin/env python3


def usage():
    print("===================================================================================================")
    print(" Quickly convert .MRC mode 12 (float16) image to .MRC mode 2 (float32) for compability with ")
    print(" programs that cannot read float16 images (i.e. CTFFIND4). ")
    print(" Usage:")
    print("    $ recast_mrc_f16_to_f32.py  input.mrc  desired/output/path ")
    print(" Example for converting an entire directory:")
    print("    $ for m in *.mrc; do recast_mrc_f16_to_f32.py $m f32/; done")
    # print(" -----------------------------------------------------------------------------------------------")
    # print(" Options (defaults in brackets): ")
    # print("        --jpg (4) : also save a (binned) .jpg image of the .MRC file")
    # print("          --j (4) : batch mode is optionally multithreaded across given # of cores")
    print("===================================================================================================")
    sys.exit()

def get_mrc_data(mrc_file):
    
    with mrcfile.open(mrc_file, permissive = True, mode = 'r') as mrc:
        mrc_data = mrc.data.astype(np.float32) 
        pixel_size = np.around(mrc.voxel_size.item(0)[0], decimals = 2)

    print(" Opened file: ")
    print("     %s" % mrc_file)
    print("     %s angpix" % pixel_size)
    return  mrc_data, pixel_size

def save_mrc_image(image_data, output_name, angpix):

    with mrcfile.new(output_name, overwrite = True) as mrc:
        mrc.set_data(image_data)
        mrc.voxel_size = angpix
        mrc.update_header_from_data()
        mrc.update_header_stats()

    print(" Saved f32 image -> ")
    print("    ", output_name)
    return 

if __name__ == "__main__":
    import mrcfile 
    import numpy as np
    import sys, os

    cmdline = sys.argv 
    min_args = 2

    if len(cmdline) - 1 < min_args:
        print(" Not enough arguments")
        usage()

    for entry in cmdline:
        if entry in ['-h', '-help', '--h', '--help']:
            usage()

    print("==================================================================")

    f = sys.argv[1]
    out_path = sys.argv[2]
    print(" Convert .MRC image from float16 -> float32")

    ## read the input image data 
    img_data, angpix = get_mrc_data(f)

    ## update the save path
    save_fname = os.path.join(out_path, os.path.split(f)[1]) 

    ## save the output image in the new format 
    save_mrc_image(img_data, save_fname, angpix)

## Example usage: 
##      $ for m in *.mrc; do ../recast_f16_to_f32.py $m float32/; done