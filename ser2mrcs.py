#!/usr/bin/env python3

"""
    A script to read in a .SER file image stack and output an .MRCS file that
    can be used with usual EM-related motion correction software.

    Typical usage is taking a 'continuous' movie with TIA and then manually
    stopping it. The output movie can then be used with this script using
    only the chosen starting frames. 
"""

## 2021-10-27: Script started
## To Do: make memory mapped version of empty mrcs, then update its values there before writing out to disk

#############################
###     FLAGS
#############################
DEBUG = True

#############################
###     DEFINITION BLOCK
#############################

def usage():
    print("===================================================================================================")
    print(" Convert an image series in .SER format to .MRCS, useable by common motion correction softwares.")
    print(" Usage:")
    print("    $ ser2mrcs.py  <input>.ser")
    print(" -----------------------------------------------------------------------------------------------")
    print(" Options: ")
    print("        --frames (n) : take only the first 'n' frames from the .SER file, e.g. --frames 5")
    print("===================================================================================================")
    sys.exit()
    return

def get_ser_data(file):
    ## use serReader module to parse the .SER file data into memory
    ## image data is returned under the key 'data' as an np.ndarray of dimension x, y from TIA .SER image
    im = serReader.serReader(file)

    for frame in im['data']:
        ## take only the first frame
        pixel_dimensions = (im['data'].shape[1], im['data'].shape[2])
        break

    number_of_frames = len(im['data'])

    if DEBUG:
        print(" ... %s total frames (%s, %s) in: %s" % (number_of_frames, pixel_dimensions[0], pixel_dimensions[1], file))

    ## recast the int32 data coming out of serReader into float32 for use as mrc mode #2
    im_float32 = im['data'].astype(np.float32)
    return im_float32, number_of_frames, pixel_dimensions

def make_empty_mrcs(stack_size, frame_dimensions, out_name):
    """ Prepare an empty .MRCS in memory of the correct dimensionality
    """
    with mrcfile.new(out_name, overwrite=True) as mrc:
        mrc.set_data(np.zeros(( stack_size, ## stack size
                                frame_dimensions[1], ## pixel height, 'Y'
                                frame_dimensions[0]  ## pixel length, 'X'
                            ), dtype=np.int16))

        ## set the mrcfile with the correct header values to indicate it is an image stack
        mrc.set_image_stack()
        if DEBUG:
            print(" ... prepared empty .MRCS (%s) to populate " % out_name, mrc.data.shape)
    return

def write_img_to_stack(mrcs_name, stack_index, frame_data):
    """
    PARAMETERS
        mrcs_name = str(); file name of the .MRCS
        stack_index = int(); position in the .MRCS stack to write/slice into
        frame_data = np.array; raw image data to write into stack at target index
    """

    with mrcfile.open(mrcs_name, mode='r+') as mrcs:
        print("================================")
        mrcs.data[stack_index] = frame_data

    if DEBUG:
        print(" >> writing frame #%s to: %s" % (stack_index, mrcs_name))

    return None

#############################
###     RUN BLOCK
#############################

if __name__ == "__main__":

    import os
    import sys
    import glob
    import numpy as np
    import time

    try:
        import mrcfile
    except:
        print(" Could not import mrcfile module. Install via:")
        print(" > pip install mrcfile")
        sys.exit()

    try:
        import serReader # serReader must be in the path as this script
    except:
        print(" Could not import serReader module. Make sure script is in same directory as main script: ")
        print("  > %s" % os.path.realpath(__file__))
        sys.exit()

    ###################################
    ## check usage
    if len(os.sys.argv) < 2:
        usage()

    ser_file = os.sys.argv[1]
    if len(ser_file) < 5:
        usage()

    if os.sys.argv[1][-4:] != ".ser":
        usage()
    ###################################

    start_time = time.time()

    output_mrcs_name = ser_file[:-4] + ".mrcs"

    ser_data, number_of_frames, frame_dimensions = get_ser_data(ser_file)

    ## allow the user to adjust the total number of frames to keep
    if '--frames' in os.sys.argv:
        ## find the index of the flag
        flag_index = os.sys.argv.index('--frames')
        ## sanity check there exists an entry next to the flag before attempting to parse it
        if len(os.sys.argv) <= flag_index + 1:
            print(" ... using default --frames value of 5")
            total_frames_allowed = 5
        else:
            try:
                total_frames_allowed = int(os.sys.argv[flag_index + 1])
            except:
                print(" ERROR :: '--frames' flag requires an integer as input (%s given)" % os.sys.argv[flag_index + 1])
                sys.exit()
            ## sanity check range provided
            if total_frames_allowed > number_of_frames or total_frames_allowed < 1:
                print(" ERROR :: incompatible values provided for '--frames' flag, must be within 1 -> %s" % number_of_frames)
                sys.exit()

        if DEBUG:
            print(" ... using only first %s frames for conversion" % total_frames_allowed)
    else:
        total_frames_allowed = number_of_frames

    empty_mrcs = make_empty_mrcs(total_frames_allowed, frame_dimensions, output_mrcs_name)

    for n in range(total_frames_allowed):
        current_frame = ser_data[n]
        write_img_to_stack(output_mrcs_name, n, current_frame)

    print(" JOB COMPLETE")
    end_time = time.time()
    total_time_taken = end_time - start_time
    print(" ... runtime = %.2f sec" % total_time_taken)
