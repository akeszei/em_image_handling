#!/usr/bin/env python3

"""
    A script to convert TIA-format .SER images to single-precision 32-bit float .MRC mode #2 format
    Serves as a replacement to EMAN e2proc2d.py conversion script
"""

## 2021-08-25: Script written
## 2021-08-28: Updated error handling and command line parsing to make more general/flexible
## 2021-09-02: Rewrite flow of execution to accomodate parallel processing

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
    print("    $ ser2mrc.py  @.mrc")
    print(" -----------------------------------------------------------------------------------------------")
    print(" Options (defaults in brackets): ")
    print("        --jpg (4) : also save a (binned) .jpg image of the .MRC file")
    print("          --j (4) : batch mode is optionally multithreaded across given # of cores")
    print("===================================================================================================")
    sys.exit()

# def read_flag(cmd_line, flag, cmd_line_flag_index, PARAMS_key, data_type, legal_entries, is_toggle, has_defaults):
#     global PARAMS, FLAGS
#     ## if the flag serves as a toggle, switch it on and exit
#     if is_toggle:
#         PARAMS[PARAMS_key] = True
#         print(" ... set: %s = %s" % (PARAMS_key, True))
#         return
#
#     ## if the flag has a default setting, quickly sanity check if we are using it
#     if has_defaults:
#         ## if there are no more entries on the command line after the flag, we necessarily are using the defaults
#         if len(sys.argv[1:]) <= cmd_line_flag_index:
#             print(" ... use default: %s = %s" % (PARAMS_key, PARAMS[PARAMS_key]))
#             return
#         else:
#             ## check if subsequent entry on cmd line is a flag itself, in which case we are using defaults
#             if cmd_line[cmd_line_flag_index + 1] in FLAGS:
#                 print(" ... use default: %s = %s" % (PARAMS_key, PARAMS[PARAMS_key]))
#                 return
#
#     ## sanity check there exists an entry next to the flag before attempting to parse it
#     if len(sys.argv[1:]) <= cmd_line_flag_index:
#         print(" ERROR :: No value provided for flag (%s)" % flag)
#         usage()
#         return
#     ## parse the entry next to the flag depending on its expected type and range
#     ## 1) INTEGERS
#     if isinstance(data_type, int):
#         try:
#             user_input = int(cmd_line[cmd_line_flag_index + 1])
#         except:
#             print(" ERROR :: %s flag requires an integer as input (%s given)" % (flag, cmd_line[cmd_line_flag_index + 1]))
#             usage()
#             return
#         ## check if the assigned value is in the expected range
#         if legal_entries[0] <= user_input <= legal_entries[1]:
#             PARAMS[PARAMS_key] = user_input
#             print(" ... set: %s = %s" % (PARAMS_key, PARAMS[PARAMS_key]))
#         else:
#             print(" ERROR :: %s flag input (%s) out of expected range: [%s, %s]" % (flag, user_input, legal_entries[0], legal_entries[1]))
#             usage()
#             return
#     ## 2) FLOATS
#     if isinstance(data_type, float):
#         try:
#             user_input = float(cmd_line[cmd_line_flag_index + 1])
#         except:
#             print(" ERROR :: %s flag requires a float as input (%s given)" % (flag, cmd_line[cmd_line_flag_index + 1]))
#             usage()
#             return
#         ## check if the assigned value is in the expected range
#         if legal_entries[0] <= user_input <= legal_entries[1]:
#             PARAMS[PARAMS_key] = user_input
#             print(" ... set: %s = %s" % (PARAMS_key, PARAMS[PARAMS_key]))
#         else:
#             print(" ERROR :: %s flag input (%s) out of expected range: [%s, %s]" % (flag, user_input, legal_entries[0], legal_entries[1]))
#             usage()
#             return
#     ## 3) STRINGS
#     if isinstance(data_type, str):
#         try:
#             user_input = cmd_line[cmd_line_flag_index + 1]
#         except:
#             print(" ERROR :: %s flag requires a string as input (%s given)" % (flag, cmd_line[cmd_line_flag_index + 1]))
#             usage()
#             return
#         ## check if the assigned value is a legal keyword
#         if user_input in legal_entries:
#             PARAMS[PARAMS_key] = user_input
#             print(" ... set: %s = %s" % (PARAMS_key, PARAMS[PARAMS_key]))
#         else:
#             print(" ERROR :: %s flag input (%s) is not a legal entry, try one of: " % (flag, user_input))
#             print(legal_entries)
#             usage()
#             return
#
# def parse_cmd_line(min_input = 1):
#     """ min_input = number of command line arguments needed at minimum to run
#     """
#     global PARAMS, FLAGS, FILES
#     ## retrieve all entries on the cmd line and parse them into global variables
#     cmd_line = tuple(sys.argv)
#
#     ## check there is a minimum number of arguments input by the user
#     if len(cmd_line) - 1 < min_input:
#         usage()
#     ## check for the help flag with elevated priority
#     for entry in cmd_line:
#         if entry in ['-h', '-help', '--h', '--help']:
#             print(' ... help flag called (%s), printing usage and exiting.' % entry)
#             usage()
#
#     ## check first if batch mode is being activated, if not then check for the proper file in each argument position
#     if '@' in os.path.splitext(cmd_line[1])[0]:
#         PARAMS['BATCH_MODE'] = True
#         print(" ... batch mode = ON")
#
#         ## if batchmode is active, then confirm the requested filetype is expected
#         if not os.path.splitext(cmd_line[1])[1] in FILES[1][1]:
#             print(" ERROR :: Requested output filetype (%s) not recognized. Try one of: %s" % (os.path.splitext(cmd_line[1])[1], FILES[1][1]))
#             sys.exit()
#
#     else:
#         for index, expected_extension, key in FILES:
#             parsed_extension = os.path.splitext(cmd_line[index])[1].lower()
#             if len(parsed_extension) == 0:
#                 print(" ERROR :: Incompatible %s file provided (%s)" % (expected_extension, cmd_line[index]))
#                 usage()
#             elif os.path.splitext(cmd_line[index])[1].lower() in expected_extension:
#                 PARAMS[key] = cmd_line[index]
#                 print(" ... %s set: %s" % (key, PARAMS[key]))
#             else:
#                 print(" ERROR :: Incompatible %s file provided (%s)" % (expected_extension, cmd_line[index]))
#                 usage()
#
#     ## after checking for help flags, try to read in all flags into global dictionary
#     for entry in cmd_line:
#         if entry in FLAGS:
#             # print("Entry found: %s (index %s)" % (entry, cmd_line.index(entry)))
#             read_flag(cmd_line, entry, cmd_line.index(entry), FLAGS[entry][0], FLAGS[entry][1], FLAGS[entry][2], FLAGS[entry][3], FLAGS[entry][4])
#         elif "--" in entry:
#             print(" WARNING : unexpected flag detected (%s), may not be correctly assigned." % entry)
#
#     return

def get_ser_data(file):
    import serReader
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

    try:
        from PIL import Image
    except:
        print(" Could not import PIL.Image. Install depenency via:")
        print(" > pip install --upgrade Pillow")
        sys.exit()

    ## sanity check the binning factor
    try:
        binning_factor = int(binning_factor)
    except:
        print("ERROR: Incompatible binnign factor provided: %s, use an integer (E.g. --jpeg 4)" % (binning_factor))

    ## open the mrc file and use its data to generate the image
    with mrcfile.open(mrc_file) as mrc:
        ## rescale the image data to grayscale range (0,255)
        remapped = (255*(mrc.data - np.min(mrc.data))/np.ptp(mrc.data)).astype(np.uint8) ## remap data from 0 -- 255
        ## load the image data into a PIL.Image object
        im = Image.fromarray(remapped).convert('L')
        ## adjust the contrast
        im = sigma_contrast(im, 3)
        ## bin the image to the desired size
        resized_im = im.resize((int(im.width/binning_factor), int(im.height/binning_factor)), Image.BILINEAR)
        jpg_name = os.path.splitext(mrc_file)[0] + '.jpg'
        # im.show()
        ## save the image to disc
        resized_im.save(jpg_name)

    if DEBUG:
        print("    >> .jpg written: %s (%s x binning)" % (jpg_name, binning_factor))

    return

def sigma_contrast(im_array, sigma):
    """ Rescale the image intensity levels to a range defined by a sigma value (the # of
        standard deviations to keep). Can perform better than auto_contrast when there is
        a lot of dark pixels throwing off the level balancing.
    """
    stdev = np.std(im_array)
    mean = np.mean(im_array)
    print("Image intensity data (mean, stdev) = (%s, %s)" % (mean, stdev))
    minval = mean - (stdev * sigma)
    maxval = mean + (stdev * sigma)
    ## remove pixles above/below the defined limits
    im_array = np.clip(im_array, minval, maxval)
    ## rescale the image into the range 0 - 255
    im_array = ((im_array - minval) / (maxval - minval)) * 255

    return im_array

def convert_image(ser_file, mrc_file, PRINT_JPEG, jpg_binning):
    """ To support parallelization, create an `execution' function that can be passed into
        multiple threads easily with only the necessary input variables
    """
    check_dependencies()
    ## get data from .SER file
    im_data = get_ser_data(ser_file)
    ## save the data to a .MRC file
    save_mrc_image(im_data, mrc_file)
    ## optionally save a .JPEG file with binning
    if PRINT_JPEG:
        save_jpeg_image(mrc_file, jpg_binning)
    return

def check_dependencies():
    ## load built-in packages, if they fail to load then python install is completely wrong!
    globals()['sys'] = __import__('sys')
    globals()['os'] = __import__('os')
    globals()['glob'] = __import__('glob')
    globals()['mp'] = __import__('multiprocessing')

    try:
        globals()['np'] = __import__('numpy') ## similar to: import numpy as np
    except:
        print(" ERROR :: Failed to import 'numpy'. Try: pip install numpy")
        sys.exit()

    try:
        from PIL import Image
    except:
        print(" Could not import PIL.Image. Install depenency via:")
        print(" > pip install --upgrade Pillow")
        sys.exit()

    try:
        globals()['mrcfile'] = __import__('mrcfile') ## similar to: import numpy as np
    except:
        print(" ERROR :: Failed to import 'mrcfile'. Try: pip install mrcfile")
        sys.exit()


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
    import time
    import cmdline_parser
    from multiprocessing import Pool
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
        'ser_file' : str(),
        'mrc_file' : str(),
        'BATCH_MODE' : False,
        'PRINT_JPEG' : False,
        'jpg_binning_factor' : 4,
        'PARALLEL_PROCESSING': False,
        'threads' : 4
        }
    ##################################

    ##################################
    ## SET UP EXPECTED DATA FOR PARSER
    ##################################
    FLAGS = {
##    flag      :  (PARAMS_key,   data_type,  legal_entries/range,  is entry a toggle,  intrinsic toggle, has_defaults)
    '--jpg'      :  ('jpg_binning_factor', int(), (1,999), False, (True, 'PRINT_JPEG', True), True),
    # '--bin_jpg'  :  ('jpg_binning_factor',   int(),     (1, 999),               False,      True ),
    '--j'        :  ('threads', int(), (1,999), False, (True, 'PARALLEL_PROCESSING', True), True)
    }

    FILES = { ## cmd line index    allowed extensions   ## can launch batch mode
        'ser_file' : (  1,         '.ser',              False),
        'mrc_file' : (  2,         '.mrc',              True)
        }
    ##################################

    start_time = time.time()

    PARAMS, EXIT_CODE = cmdline_parser.parse(sys.argv, 1, PARAMS, FLAGS, FILES)
    if EXIT_CODE < 0:
        # print("Could not correctly parse cmd line")
        usage()
        sys.exit()
    cmdline_parser.print_parameters(PARAMS, sys.argv)


    ## add a custom checks outside scope of general parser above
    commands = []
    ## get all commands used
    for n in range(len(sys.argv[1:])+1):
        commands.append(sys.argv[n])

    ## single image conversion mode
    if not PARAMS['BATCH_MODE']:

        convert_image(PARAMS['ser_file'], PARAMS['mrc_file'], PARAMS['PRINT_JPEG'], PARAMS['jpg_binning_factor'])

    ## batch image conversion mode
    else:
        if PARAMS['PARALLEL_PROCESSING']:
            ## permit multithreading
            threads = PARAMS['threads']
            print(" ... multithreading activated (%s threads) " % threads)

            ## multithreading set up
            tasks = []
            for file in glob.glob("*.ser"):
                tasks.append(file) ## inputs to the target function

            try:
                ## assign inputs for full dataset
                dataset = []
                for task in tasks:
                    ser_file = task
                    mrc_file = os.path.splitext(task)[0] + ".mrc"
                    dataset.append((ser_file, mrc_file, PARAMS['PRINT_JPEG'], PARAMS['jpg_binning_factor']))

                ## prepare pool of workers
                pool = Pool(threads)
                ## assign workload to pool
                results = pool.starmap(convert_image, dataset)
                ## close the pool from recieving any other tasks
                pool.close()
                ## merge with the main thread, stopping any further processing until workers are complete
                pool.join()

            except KeyboardInterrupt:
                print("Multiprocessing run killed")
                pool.terminate()
        else:
            ## get all files with extension
            for file in glob.glob("*.ser"):
                ## then run through them one-by-one
                PARAMS['ser_file'] = file
                current_file_base_name = os.path.splitext(file)[0]
                PARAMS['mrc_file'] = current_file_base_name + ".mrc"
                convert_image(PARAMS['ser_file'], PARAMS['mrc_file'], PARAMS['PRINT_JPEG'], PARAMS['jpg_binning_factor'])
                #
                # ## get data from .SER file
                # im_data = get_ser_data(PARAMS['ser_file'])
                # ## save the data to a .MRC file
                # save_mrc_image(im_data, PARAMS['mrc_file'])
                # ## optionally save a .JPEG file with binning
                # if PARAMS['PRINT_JPEG']:
                #     save_jpeg_image(PARAMS['mrc_file'], PARAMS['jpg_binning_factor'])

                # current_file_base_name = os.path.splitext(file)[0]
                # ## get data from .SER file
                # im_data = get_ser_data(file)
                # ## save the data to a .MRC file
                # save_mrc_image(im_data, current_file_base_name + ".mrc")
                # ## optionally save a .JPEG file with binning
                # if PARAMS['PRINT_JPEG']:
                #     save_jpeg_image(current_file_base_name + ".mrc", PARAMS['jpg_binning_factor'])


    end_time = time.time()
    total_time_taken = end_time - start_time
    print("Total time taken to run = %.2f sec" % total_time_taken)
    ## non-parallelized, 4 imgs = ~6.25 sec
    ## parallized, pool mode, 4 imgs 4 threads = ~ 4 sec
