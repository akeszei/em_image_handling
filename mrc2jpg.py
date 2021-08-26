#!/usr/bin/env python3

"""
    A script to convert .MRC images to .JPG (or other) format with optional binning
    and scalebar.
"""

## 2021-08-26: Script written

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
    print(" Convert .MRC 2D EM images to .JPG format. Alternative output formats can be used (.PNG, .TIF, .GIF)")
    print(" Options include binning and addition of a scalebar of specified size.")
    print(" Usage:")
    print("    $ mrc2jpg.py  input.mrc  output.jpg")
    print(" Batch mode:")
    print("    $ ser2mrc.py  *.mrc  @.jpg")
    print(" -----------------------------------------------------------------------------------------------")
    print(" Options: ")
    print("           --bin (4) : binning factor for image")
    print("    --scalebar (200) : add scalebar in Angstroms. Note: uses 1.94 Ang/px by default")
    print("     --angpix (1.94) : Angstroms per pixel in .mrc image")
    print("===================================================================================================")
    sys.exit()
    return

def parse_flags():
    global BATCH_MODE, mrc_file, output_file, BIN_IMAGE, binning_factor, PRINT_SCALEBAR, scalebar_angstroms, angpix

    ## a list of all flags expected by the parser, necessary to allow default (i.e. unassigned flag) behaviour
    EXPECTED_FLAGS = ['--bin', '--scalebar', '--angpix']

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
        if os.path.splitext(sys.argv[n])[1].lower() in ['.jpg', '.jpeg', '.png', '.tif', '.gif']:
            output_file = sys.argv[n]
        ## read any entry with '.mrc' as the .MRC file
        if os.path.splitext(sys.argv[n])[1].lower() == '.mrc':
            mrc_file = sys.argv[n]
        if sys.argv[n] == '--scalebar':
            PRINT_SCALEBAR = True
            ## check in case it is the last item on the cmd line and no input was given (i.e. use defaults)
            if len(sys.argv[1:]) > n:
                 ## check if the next element is also an option or an expected input file (i.e. no value provided to this option, use its defaults)
                 if sys.argv[n+1] in EXPECTED_FLAGS:#== '--scalebar' or sys.argv[n+1] == '--angpix':
                     pass
                 elif os.path.splitext(sys.argv[n])[1].lower() in ['.mrc', '.jpg', '.jpeg', '.png', '.tif', '.gif']:
                     pass
                 else:
                    try:
                        scalebar_angstroms = int(sys.argv[n+1])
                    except ValueError:
                        print("ERROR: Incompatible entry given for --scalebar option: %s ; use a positive integer" % sys.argv[n+1])
                        usage()
            if scalebar_angstroms < 0:
                print("ERROR: Incompatible entry given for --scalebar option: %s ; use a positive integer" % sys.argv[n+1])
                usage()
        if sys.argv[n] == '--bin':
            BIN_IMAGE = True
            ## check in case it is the last item on the cmd line and no input was given (i.e. use defaults)
            if len(sys.argv[1:]) > n:
                ## check if the next element is also an option or an expected input file (i.e. no value provided to this option, use its defaults)
                if sys.argv[n+1] in EXPECTED_FLAGS:#== '--scalebar' or sys.argv[n+1] == '--angpix':
                    pass
                elif os.path.splitext(sys.argv[n])[1].lower() in ['.mrc', '.jpg', '.jpeg', '.png', '.tif', '.gif']:
                    pass
                else:
                    try:
                        binning_factor = int(sys.argv[n+1])
                    except ValueError:
                        print("ERROR: Incompatible entry given for --bin option: %s ; use a positive integer" % sys.argv[n+1])
                        usage()

            if binning_factor <= 0:
                print("ERROR: Incompatible entry given for --bin option: %s ; use a positive integer" % sys.argv[n+1])
                usage()
        if sys.argv[n] == '--angpix':
            ## check in case it is the last item on the cmd line and no input was given (i.e. use defaults)
            if len(sys.argv[1:]) > n:
                if sys.argv[n+1] in EXPECTED_FLAGS:#== '--scalebar' or sys.argv[n+1] == '--angpix':
                    pass
                elif os.path.splitext(sys.argv[n])[1].lower() in ['.mrc', '.jpg', '.jpeg', '.png', '.tif', '.gif']:
                    pass
                else:
                    try:
                        angpix = float(sys.argv[n+1])
                    except ValueError:
                        print("ERROR: Incompatible --angpix flag given, must be float")
                        usage()



    ## parse if user is attempting to run in batch mode
    if "*.mrc" in mrc_file:
        for elem in ['@.jpg', '@.jpeg', '@.png', '@.tif', '@.gif']:
            if elem in output_file:
                BATCH_MODE = True
        ## if we have parsed the output file and not recieved a go ahead for batch mode, throw an error
        if not BATCH_MODE:
            print("ERROR: Batch mode via *.mrc given as output, but incorrect output provided: %s, try: @.jpg" % output_file)
            usage()

    ## check we have minimal input necessary to run
    if len(mrc_file) <= 0:
        print(" ERROR: missing an assigned .MRC file")
        usage()
    elif len(output_file) <= 0:
        print(" ERROR: missing an assigned .SER file")
        usage()

    if not ".mrc" in mrc_file:
        print(" ERROR: missing an assigned .MRC file")
        usage()
    if not os.path.splitext(output_file)[1].lower() in ['.mrc', '.jpg', '.jpeg', '.png', '.tif', '.gif']:
        print(" ERROR: missing an assigned output file")
        usage()

    ## print warning if no --angpix is given but --scalebar is (i.e. user may want to use a differnet pixel size)
    if PRINT_SCALEBAR:
        commands = []
        ## get all commands
        for n in range(len(sys.argv[1:])+1):
            commands.append(sys.argv[n])
        ## check if --angpix was given
        if not '--angpix' in commands:
            print("!! WARNING: --scalebar was given without an explicit --angpix, using default value of 1.94 Ang/px !!")


    if DEBUG:
        print(" Parsed input flags:")
        if BATCH_MODE:
            print("  ... Batch mode ON")
            print("     > Output format: %s" % os.path.splitext(output_file)[1].lower() )
        else:
            print("  ... Input file, Output file = (%s, %s)" % (mrc_file, output_file))
        if BIN_IMAGE:
            print("  ... Binning = %s" % binning_factor)
        if PRINT_SCALEBAR:
            print("  ... Add scalebar (%s px, %s Angstroms, %s Ang/pix)" % (int(scalebar_angstroms/angpix), scalebar_angstroms, angpix))
    return

def get_mrc_data(file):
    """ file = .mrc file
        returns np.ndarray of the mrc data using mrcfile module
    """
    with mrcfile.open(file) as mrc:
        image_data = mrc.data
    return image_data

def save_image(mrc_data, mrc_filename, output_file, BATCH_MODE, binning_factor, PRINT_SCALEBAR, scalebar_angstroms, angpix):

    ## rescale the image data to grayscale range (0,255)
    remapped = (255*(mrc_data - np.min(mrc_data))/np.ptp(mrc_data)).astype(int) ## remap data from 0 -- 255
    ## load the image data into a PIL.Image object
    im = Image.fromarray(remapped).convert('RGB')

    ## figure out the name of the file
    if BATCH_MODE:
        ## in batch mode, inherit the base name of the .MRC file, just change the extension
        output_format = os.path.splitext(output_file)[1].lower()
        img_name = os.path.splitext(mrc_filename)[0] + output_format
    else:
        img_name = output_file

    ## bin the image to the desired size
    resized_im = im.resize((int(im.width/binning_factor), int(im.height/binning_factor)), Image.BILINEAR)

    # make a scalebar if requested
    if PRINT_SCALEBAR:
        rescaled_angpix = angpix * binning_factor
        scalebar_px = int(scalebar_angstroms / rescaled_angpix)
        resized_im = add_scalebar(resized_im, scalebar_px)

    ## save the image to disc
    resized_im.save(img_name)
    # resized_im.show()

    if DEBUG:
        print("  >> .jpg written: %s" % img_name )

    return

def add_scalebar(image_obj, scalebar_px):
    """ Adds a scalebar to the input image and returns a new edited image
    """
    ## set the indentation to be ~2.5% inset from the bottom left corner of the image
    indent_px = int(image_obj.height * 0.025)
    ## set the stroke to be ~0.5% image size
    stroke = int(image_obj.height * 0.005)
    if stroke < 1:
        stroke = 1
    print("Scale bar info: (offset px, stroke) = (%s, %s)" % (indent_px, stroke))
    ## find the pixel range for the scalebar, typically 5 x 5 pixels up from bottom left
    LEFT_INDENT = indent_px # px from left to indent the scalebar
    BOTTOM_INDENT = indent_px # px from bottom to indent the scalebar
    STROKE = stroke # px thickness of scalebar
    x_range = (LEFT_INDENT, LEFT_INDENT + scalebar_px)
    y_range = (image_obj.height - BOTTOM_INDENT - STROKE, image_obj.height - BOTTOM_INDENT)

    ## set the pixels white for the scalebar
    for x in range(x_range[0], x_range[1]):
        for y in range(y_range[0], y_range[1]):
            image_obj.putpixel((x, y), (255, 255, 255))
    return image_obj

#############################
###     RUN BLOCK
#############################

if __name__ == "__main__":

    import os
    import sys
    import glob
    from PIL import Image
    import numpy as np
    import mrcfile

    ##################################
    ## DEFAULT VARIABLES
    ##################################
    BATCH_MODE = False
    output_file = ''
    mrc_file = ''
    BIN_IMAGE = False
    binning_factor = 4
    PRINT_SCALEBAR = False
    scalebar_angstroms = 200 # Angstroms
    angpix = 1.94
    ##################################

    ## read input
    parse_flags()

    if not BATCH_MODE:
        ## single image conversion mode
        save_image(get_mrc_data(mrc_file), mrc_file, output_file, BATCH_MODE, binning_factor, PRINT_SCALEBAR, scalebar_angstroms, angpix)
    else:
        ## get all files with extension
        for file in glob.glob("*.mrc"):
            save_image(get_mrc_data(file), file, output_file, BATCH_MODE, binning_factor, PRINT_SCALEBAR, scalebar_angstroms, angpix)
