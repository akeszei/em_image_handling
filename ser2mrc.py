#!/usr/bin/env python3

"""
    A script to convert TIA-format .SER images to single-precision 32-bit float .MRC mode #2 format
    Serves as a replacement to EMAN e2proc2d.py conversion script
"""

## 2021-08-25: Script written

#############################
###     FLAGS
#############################
DEBUG = True

#############################
###     DEFINITION BLOCK
#############################

def usage():
    """ This script requires several input arguments to work correctly. Check they exist, otherwise print usage and exit.
    """
    print("===================================================================================================")
    print(" Script to convert ThermoFischer TIA-generated .SER format 2D EM images to .MRC format. ")
    print(" Optionally, can output binned .jpeg images as it runs. Dependencies include mrcfile and ")
    print(" serReader.py scipt, which must be in the same directory as this script.")
    print(" Usage:")
    print("    $ ser2mrc.py  input.ser  output.mrc")
    print(" Batch mode:")
    print("    $ ser2mrc.py  *.ser  @.mrc")
    print(" -----------------------------------------------------------------------------------------------")
    print(" Options: ")
    print("    --jpeg (4) : save a .jpg image of the .MRC file, where integer given is the binning factor")
    print("===================================================================================================")
    sys.exit()

def parse_flags():
    global BATCH_MODE, mrc_file, ser_file, PRINT_JPEG, binning_factor

    ## check there is a minimum number of entries otherwise print usage and exit
    if len(sys.argv) < 3:
        usage()

    ## read all entries and check if the help flag is called at any point
    for n in range(len(sys.argv[1:])+1):
        # if the help flag is called, pring usage and exit
        if sys.argv[n] == '-h' or sys.argv[n] == '--help' or sys.argv[n] == '--h':
            usage()

    ## read all legal commandline arguments
    for n in range(len(sys.argv[1:])+1):
        ## read any entry with '.ser' as the TIA .SER file
        if os.path.splitext(sys.argv[n])[1].lower() == '.ser':
            ser_file = sys.argv[n].lower()
        ## read any entry with '.mrc' as the .MRC file
        if os.path.splitext(sys.argv[n])[1].lower() == '.mrc':
            mrc_file = sys.argv[n].lower()
        if sys.argv[n] == '--jpeg':
            PRINT_JPEG = True
            if len(sys.argv[1:]) > n:
                try:
                    binning_factor = int(sys.argv[n+1])
                except ValueError:
                    pass


    ## parse if user is attempting to run in batch mode
    if "@.mrc" in mrc_file:
        if ser_file == "*.ser":
            BATCH_MODE = True
        else:
            print("ERROR: @.mrc given as output, but incorrect input string: %s, try: *.ser" % ser_file)
            usage()

    ## check we have minimal input necessary to run
    if not ".mrc" in mrc_file:
        print(" ERROR: missing an assigned .MRC file")
        usage()
    if not ".ser" in ser_file:
        print(" ERROR: missing an assigned .SER file")
        usage()

    return

def get_ser_data(file):
    ## use serReader module to parse the .SER file data into memory
    im = serReader.serReader(file)
    ## get the image data as an np.ndarray of dimension x, y from TIA .SER image
    im_data = im['data']
    ## recast the int32 data coming out of serReader into float32 for use as mrc mode #2
    im_float32 = im_data.astype(np.float32)
    return im_float32

def save_mrc_image(im_data, output_name):
    """
        im_data = needs to be np.array float32 with dimensions of the TIA image
        output_name = string; name of the output file to be saved
    """
    with mrcfile.new(output_name, overwrite = True) as mrc:
        mrc.set_data(im_data)
        mrc.update_header_from_data()
        mrc.update_header_stats()

    if DEBUG:
        print(" ... .ser converted to: %s" % output_name)
    return

def save_jpeg_image(mrc_file, binning_factor):
    ## sanity check the binning factor
    try:
        binning_factor = int(binning_factor)
    except:
        print("ERROR: Incompatible binnign factor provided: %s, use an integer (E.g. --jpeg 4)" % (binning_factor))

    ## open the mrc file and use its data to generate the image
    with mrcfile.open(mrc_file) as mrc:
        ## rescale the image data to grayscale range (0,255)
        remapped = (255*(mrc.data - np.min(mrc.data))/np.ptp(mrc.data)).astype(int) ## remap data from 0 -- 255
        ## load the image data into a PIL.Image object
        im = Image.fromarray(remapped).convert('RGB')
        ## bin the image to the desired size
        resized_im = im.resize((int(im.width/binning_factor), int(im.height/binning_factor)), Image.BILINEAR)
        jpg_name = os.path.splitext(mrc_file)[0] + '.jpg'
        # im.show()
        ## save the image to disc
        resized_im.save(jpg_name)

    if DEBUG:
        print("    >> .jpg written: %s (%s x binning)" % (jpg_name, binning_factor))

    return


#############################
###     RUN BLOCK
#############################

if __name__ == "__main__":

    # from pathlib import Path # requires python 3.4+
    # import string
    import os
    import sys
    import glob
    from PIL import Image
    import numpy as np
    import mrcfile
    import serReader # serReader must be in the path as this script

    ##################################
    ## DEFAULT VARIABLES
    ##################################
    BATCH_MODE = False
    ser_file = ''
    mrc_file = ''
    PRINT_JPEG = False
    binning_factor = 4
    ##################################

    ## read input
    parse_flags()

    if not BATCH_MODE:
        ## single image conversion mode

        ## get data from .SER file
        im_data = get_ser_data(ser_file)
        ## save the data to a .MRC file
        save_mrc_image(im_data, mrc_file)
        ## optionally save a .JPEG file with binning
        if PRINT_JPEG:
            save_jpeg_image(mrc_file, binning_factor)

    else:
        ## get all files with extension
        for file in glob.glob("*.ser"):
            ## then run through them one-by-one
            current_file_base_name = s.path.splitext(sys.argv[n])[0]
            ## get data from .SER file
            im_data = get_ser_data(file)
            ## save the data to a .MRC file
            save_mrc_image(im_data, current_file_base_name + ".mrc")
            ## optionally save a .JPEG file with binning
            if PRINT_JPEG:
                save_jpeg_image(current_file_base_name + ".mrc", binning_factor)
