#!/usr/bin/env python3

"""
    A script to convert TIA-format .SER images to single-precision 32-bit float .MRC mode #2 format
    Serves as a replacement to EMAN e2proc2d.py conversion script
"""

## 2021-08-25: Script written
## 2021-08-28: Updated error handling and command line parsing to make more general/flexible

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
    print("    $ ser2mrc.py  input.ser  output.mrc  <options>")
    print(" Batch mode:")
    print("    $ ser2mrc.py  *.ser  @.mrc")
    print(" -----------------------------------------------------------------------------------------------")
    print(" Options: ")
    print("            --jpg : also save a .jpg image of the .MRC file")
    print("    --bin_jpg (4) : bin the jpg file before saving it to disk")
    print("===================================================================================================")
    sys.exit()

def read_flag(cmd_line, flag, cmd_line_flag_index, GLOBAL_VARS_key, data_type, legal_entries, is_toggle, has_defaults):
    global GLOBAL_VARS, EXPECTED_FLAGS
    ## if the flag serves as a toggle, switch it on and exit
    if is_toggle:
        GLOBAL_VARS[GLOBAL_VARS_key] = True
        print(" ... set: %s = %s" % (GLOBAL_VARS_key, True))
        return

    ## if the flag has a default setting, quickly sanity check if we are using it
    if has_defaults:
        ## if there are no more entries on the command line after the flag, we necessarily are using the defaults
        if len(sys.argv[1:]) <= cmd_line_flag_index:
            print(" ... use default: %s = %s" % (GLOBAL_VARS_key, GLOBAL_VARS[GLOBAL_VARS_key]))
            return
        else:
            ## check if subsequent entry on cmd line is a flag itself, in which case we are using defaults
            if cmd_line[cmd_line_flag_index + 1] in EXPECTED_FLAGS:
                print(" ... use default: %s = %s" % (GLOBAL_VARS_key, GLOBAL_VARS[GLOBAL_VARS_key]))
                return

    ## sanity check there exists an entry next to the flag before attempting to parse it
    if len(sys.argv[1:]) <= cmd_line_flag_index:
        print(" ERROR :: No value provided for flag (%s)" % flag)
        usage()
        return
    ## parse the entry next to the flag depending on its expected type and range
    ## 1) INTEGERS
    if isinstance(data_type, int):
        try:
            user_input = int(cmd_line[cmd_line_flag_index + 1])
        except:
            print(" ERROR :: %s flag requires an integer as input (%s given)" % (flag, cmd_line[cmd_line_flag_index + 1]))
            usage()
            return
        ## check if the assigned value is in the expected range
        if legal_entries[0] <= user_input <= legal_entries[1]:
            GLOBAL_VARS[GLOBAL_VARS_key] = user_input
            print(" ... set: %s = %s" % (GLOBAL_VARS_key, GLOBAL_VARS[GLOBAL_VARS_key]))
        else:
            print(" ERROR :: %s flag input (%s) out of expected range: [%s, %s]" % (flag, user_input, legal_entries[0], legal_entries[1]))
            usage()
            return
    ## 2) FLOATS
    if isinstance(data_type, float):
        try:
            user_input = float(cmd_line[cmd_line_flag_index + 1])
        except:
            print(" ERROR :: %s flag requires a float as input (%s given)" % (flag, cmd_line[cmd_line_flag_index + 1]))
            usage()
            return
        ## check if the assigned value is in the expected range
        if legal_entries[0] <= user_input <= legal_entries[1]:
            GLOBAL_VARS[GLOBAL_VARS_key] = user_input
            print(" ... set: %s = %s" % (GLOBAL_VARS_key, GLOBAL_VARS[GLOBAL_VARS_key]))
        else:
            print(" ERROR :: %s flag input (%s) out of expected range: [%s, %s]" % (flag, user_input, legal_entries[0], legal_entries[1]))
            usage()
            return
    ## 3) STRINGS
    if isinstance(data_type, str):
        try:
            user_input = cmd_line[cmd_line_flag_index + 1]
        except:
            print(" ERROR :: %s flag requires a string as input (%s given)" % (flag, cmd_line[cmd_line_flag_index + 1]))
            usage()
            return
        ## check if the assigned value is a legal keyword
        if user_input in legal_entries:
            GLOBAL_VARS[GLOBAL_VARS_key] = user_input
            print(" ... set: %s = %s" % (GLOBAL_VARS_key, GLOBAL_VARS[GLOBAL_VARS_key]))
        else:
            print(" ERROR :: %s flag input (%s) is not a legal entry, try one of: " % (flag, user_input))
            print(legal_entries)
            usage()
            return

def parse_cmd_line():
    global GLOBAL_VARS, EXPECTED_FLAGS, EXPECTED_FILES
    ## retrieve all entries on the cmd line and parse them into global variables
    cmd_line = tuple(sys.argv)
    ## check for the help flag with elevated priority
    for entry in cmd_line:
        if entry in ['-h', '-help', '--h', '--help']:
            print(' ... help flag called (%s), printing usage and exiting.' % entry)
            usage()

    ## load all expected files based on their explicit index and check for proper extension in name
    for file in EXPECTED_FILES:
        if os.path.splitext(cmd_line[file[0]])[1].lower() == file[1]:
            GLOBAL_VARS[file[2]] = cmd_line[file[0]]
            print(" ... %s set: %s" % (file[2], GLOBAL_VARS[file[2]]))
        else:
            print(" ERROR :: Incompatible %s file provided (%s)" % (file[1], cmd_line[file[0]]))
            usage()

    ## determine if batch mode is being attempted
    if "@.mrc" in GLOBAL_VARS['mrc_file']:
        if GLOBAL_VARS['ser_file'] == "*.ser":
            GLOBAL_VARS['BATCH_MODE'] = True
            print(" ... batch mode = ON")
        else:
            print("ERROR :: Batch mode detected (@.mrc), but incorrect .ser file entry (%s), try: *.ser" % GLOBAL_VARS['ser_file'])
            usage()

    ## after checking for help flags, try to read in all flags into global dictionary
    for entry in cmd_line:
        if entry in EXPECTED_FLAGS:
            # print("Entry found: %s (index %s)" % (entry, cmd_line.index(entry)))
            read_flag(cmd_line, entry, cmd_line.index(entry), EXPECTED_FLAGS[entry][0], EXPECTED_FLAGS[entry][1], EXPECTED_FLAGS[entry][2], EXPECTED_FLAGS[entry][3], EXPECTED_FLAGS[entry][4])
        elif "--" in entry:
            print(" WARNING : unexpected flag detected (%s), may not be correctly assigned." % entry)

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

    ##################################
    ## DEPENDENCIES
    ##################################
    import os
    import sys
    import glob
    import numpy as np

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
    GLOBAL_VARS = {
        'ser_file' : str(),
        'mrc_file' : str(),
        'BATCH_MODE' : False,
        'PRINT_JPEG' : False,
        'jpg_binning_factor' : 4
        }
    ##################################

    ##################################
    ## SET UP EXPECTED DATA FOR PARSER
    ##################################
    EXPECTED_FLAGS = {
     ##    flag      :  (GLOBAL_VARS_key,   data_type,  legal_entries/range,    is_toggle,   has_defaults)
        '--jpg'      :  ('PRINT_JPEG'   ,    bool(),    (),                     True,       False ),
        '--bin_jpg'  :  ('jpg_binning_factor',   int(),     (1, 999),            False,      True )
    }

    EXPECTED_FILES = [  # cmd_line_index,   expected_extension,     GLOBAL_VARS_key
                    (   1,                  '.ser',                 'ser_file'),
                    (   2,                  '.mrc',                 'mrc_file')
                    ]
    ##################################

    parse_cmd_line()

    ## single image conversion mode
    if not GLOBAL_VARS['BATCH_MODE']:

        ## get data from .SER file
        im_data = get_ser_data(GLOBAL_VARS['ser_file'])
        ## save the data to a .MRC file
        save_mrc_image(im_data, GLOBAL_VARS['mrc_file'])
        ## optionally save a .JPEG file with binning
        if GLOBAL_VARS['PRINT_JPEG']:
            save_jpeg_image(GLOBAL_VARS['mrc_file'], GLOBAL_VARS['jpg_binning_factor'])

    ## batch image conversion mode
    else:
        ## get all files with extension
        for file in glob.glob("*.ser"):
            ## then run through them one-by-one
            current_file_base_name = os.path.splitext(file)[0]
            ## get data from .SER file
            im_data = get_ser_data(file)
            ## save the data to a .MRC file
            save_mrc_image(im_data, current_file_base_name + ".mrc")
            ## optionally save a .JPEG file with binning
            if GLOBAL_VARS['PRINT_JPEG']:
                save_jpeg_image(current_file_base_name + ".mrc", GLOBAL_VARS['jpg_binning_factor'])
