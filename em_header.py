#!/usr/bin/env python3

"""
    A script to get header information from common EM image formats without needing
    EMAN2 or other programs installed.
"""

## 2021-09-07: Script written

#############################
###     FLAGS
#############################
DEBUG = True

#############################
###     DEFINITION BLOCK
#############################

def usage():
    print("===================================================================================================")
    print(" Script to retrieve and print header information for common EM-formats. ")
    print(" Usage:")
    print("    $ em_header.py  <.mrc or .ser file>  <options>")
    print(" -----------------------------------------------------------------------------------------------")
    print(" Options: ")
    print("        --angpix_only : only print the Angstroms/pixel value in the header")
    print("           --dim_only : only print the dimensions of the image")
    print("===================================================================================================")
    sys.exit()

def get_ser_data(file, HEADER_INFO):
    ## use serReader module to parse the .SER file data into memory
    ser = serReader.serReader(file)
    ## get the image data as an np.ndarray of dimension x, y from TIA .SER image

    HEADER_INFO['image_dimensions'] = ser['data'].shape
    HEADER_INFO['angpix'] = "{0:.4g}".format(ser['pixelSizeX'] * 10**10)
    HEADER_INFO['filename'] = file
    HEADER_INFO['min'] = ser['data'].min()
    HEADER_INFO['max'] = ser['data'].max()
    HEADER_INFO['mean'] = ser['data'].mean()


def get_mrc_data(file, HEADER_INFO):
    with mrcfile.open(file) as mrc:
        HEADER_INFO['image_dimensions'] = '(%s, %s, %s)' % (mrc.header.nx, mrc.header.ny, mrc.header.nz)
        HEADER_INFO['filename'] = file
        HEADER_INFO['angpix'] = mrc.voxel_size.x
        HEADER_INFO['min'] = mrc.header.dmin
        HEADER_INFO['max'] = mrc.header.dmax
        HEADER_INFO['mean'] = mrc.header.dmean


        print("MRC mode = ", mrc.header.mode)
        # mrc.print_header()


def print_header(HEADER_INFO, PARAMS):
    for key in HEADER_INFO:
        if PARAMS['ANGPIX_ONLY']:
            if key == 'angpix':
                print("  %s = %s" % (key, HEADER_INFO[key]))
        if PARAMS['DIMENSIONS_ONLY']:
            if key == 'image_dimensions':
                print("  %s = %s" % (key, HEADER_INFO[key]))
        else:
            print("  %s = %s" % (key, HEADER_INFO[key]))


#############################
###     RUN BLOCK
#############################

if __name__ == "__main__":

    ##################################
    ## DEPENDENCIES
    ##################################
    import os
    import sys
    import numpy as np
    import cmdline_parser

    try:
        from PIL import Image
    except:
        print(" Could not import PIL.Image. Install depenency via:")
        print(" > pip install --upgrade Pillow")
        sys.exit()

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
    ##################################

    ##################################
    ## ASSIGN DEFAULT VARIABLES
    ##################################
    PARAMS = {
        'ANGPIX_ONLY'       : False,
        'DIMENSIONS_ONLY'   : False,
        }
    ##################################

    ##################################
    ## SET UP EXPECTED DATA FOR PARSER
    ##################################
    FLAGS = {
    '--angpix_only' : (
        'ANGPIX_ONLY', ## PARAMS key
        bool(), ## data type
        (), ## legal entries/range
        False, ## toggle information for any entries following flag
        (True, 'ANGPIX_ONLY', True), ## if flag itself has toggle information
        True ## if flag has a default setting
    ),
    '--dim_only' : (
        'DIMENSIONS_ONLY', ## PARAMS key
        bool(), ## data type
        (), ## legal entries/range
        False, ## toggle information for any entries following flag
        (True, 'DIMENSIONS_ONLY', True), ## if flag itself has toggle information
        True ## if flag has a default setting
    )
    }

    FILES = { ## cmd line index    allowed extensions   ## can launch batch mode
        'input_file' : (  -1,         ['.ser','.mrc'],  False)
        }
    ##################################

    PARAMS, EXIT_CODE = cmdline_parser.parse(sys.argv, 1, PARAMS, FLAGS, FILES)
    if EXIT_CODE < 0:
        # print("Could not correctly parse cmd line")
        usage()
        sys.exit()
    cmdline_parser.print_parameters(PARAMS, sys.argv)

    ## prepare a general header dictionary we want to populate
    HEADER_INFO = {
        'filename' : '',
        'angpix' : '',
        'image_dimensions' : '',
        'min' : '',
        'max' : '',
        'mean': ''
    }

    ## determine which filetype was submitted and pass it to the appropriate function for parsing
    extension = os.path.splitext(PARAMS['input_file'])[1]
    if extension == '.mrc':
        print("MRC submitted")
        get_mrc_data(PARAMS['input_file'], HEADER_INFO)
    elif extension == '.ser':
        get_ser_data(PARAMS['input_file'], HEADER_INFO)
        print("SER submitted")


    print_header(HEADER_INFO, PARAMS)
