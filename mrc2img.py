#!/usr/bin/env python3

"""
    A script to convert .MRC images to .JPG (or other) format with optional binning
    and scalebar.
"""

## 2021-08-26: Script written
## 2021-08-28: Updated command line parsing to make more general, and renamed script to reflect multiple image handling

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
    print(" Convert .MRC 2D EM images to conventional image formats (.PNG, .TIF, .GIF).")
    print(" Options include binning and addition of a scalebar of specified size.")
    print(" Usage:")
    print("    $ mrc2img.py  input.mrc  output.jpg/png/tif/gif  <options> ")
    print(" Batch mode:")
    print("    $ mrc2img.py  @.jpg/png/tif/gif")
    print(" -----------------------------------------------------------------------------------------------")
    print(" Options (default in brackets): ")
    print("           --bin (4) : binning factor for image")
    print("    --scalebar (200) : add scalebar in Angstroms. Note: uses 1.94 Ang/px by default")
    print("     --angpix (1.94) : Angstroms per pixel in .mrc image")
    print("             --j (4) : Allow multiprocessing using indicated number of cores")
    print("===================================================================================================")
    sys.exit()
    return

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

def parse_cmd_line(min_input = 1):
    """ min_input = number of command line arguments needed at minimum to run
    """
    global GLOBAL_VARS, EXPECTED_FLAGS, EXPECTED_FILES
    ## retrieve all entries on the cmd line and parse them into global variables
    cmd_line = tuple(sys.argv)
    ## check there is a minimum number of arguments input by the user
    if len(cmd_line) - 1 < min_input:
        usage()
    ## check for the help flag with elevated priority
    for entry in cmd_line:
        if entry in ['-h', '-help', '--h', '--help']:
            print(' ... help flag called (%s), printing usage and exiting.' % entry)
            usage()

    ## check first if batch mode is being activated, if not then check for the proper file in each argument position
    if '@' in os.path.splitext(cmd_line[1])[0]:
        GLOBAL_VARS['BATCH_MODE'] = True
        print(" ... batch mode = ON")

        ## if batchmode is active, then confirm the requested filetype is expected
        if not os.path.splitext(cmd_line[1])[1] in EXPECTED_FILES[1][1]:
            print(" ERROR :: Requested output filetype (%s) not recognized. Try one of: %s" % (os.path.splitext(cmd_line[1])[1], EXPECTED_FILES[1][1]))
            sys.exit()

    else:
        for index, expected_extension, key in EXPECTED_FILES:
            parsed_extension = os.path.splitext(cmd_line[index])[1].lower()
            if len(parsed_extension) == 0:
                print(" ERROR :: Incompatible %s file provided (%s)" % (expected_extension, cmd_line[index]))
                usage()
            elif os.path.splitext(cmd_line[index])[1].lower() in expected_extension:
                GLOBAL_VARS[key] = cmd_line[index]
                print(" ... %s set: %s" % (key, GLOBAL_VARS[key]))
            else:
                print(" ERROR :: Incompatible %s file provided (%s)" % (expected_extension, cmd_line[index]))
                usage()

    ## after checking for help flags, try to read in all flags into global dictionary
    for entry in cmd_line:
        if entry in EXPECTED_FLAGS:
            # print("Entry found: %s (index %s)" % (entry, cmd_line.index(entry)))
            read_flag(cmd_line, entry, cmd_line.index(entry), EXPECTED_FLAGS[entry][0], EXPECTED_FLAGS[entry][1], EXPECTED_FLAGS[entry][2], EXPECTED_FLAGS[entry][3], EXPECTED_FLAGS[entry][4])
        elif "--" in entry:
            print(" WARNING : unexpected flag detected (%s), may not be correctly assigned." % entry)

    return

def get_mrc_data(file):
    """ file = .mrc file
        returns np.ndarray of the mrc data using mrcfile module
    """
    with mrcfile.open(file) as mrc:
        image_data = mrc.data
    return image_data

def apply_sigma_contrast(im_data, sigma_value):
    """
        Apply sigma contrast to an image.
    PARAMETERS
        im_data = 2d np.array
        sigma_value = float; typical values are 3 - 4
    RETURNS
        2d np.array, where values from original array are rescaled based on the sigma contrast value
    """
    ## 1. find the standard deviation of the data set
    im_stdev = np.std(im_data)
    ## 2. find the mean of the dataset
    im_mean = np.mean(im_data)
    ## 3. define the upper and lower limit of the image using a chosen sigma contrast value
    sigma_contrast = 3.5 ## set the limits of the pixel values we want for min and max
    min = im_mean - (sigma_contrast * im_stdev)
    max = im_mean + (sigma_contrast * im_stdev)
    ## 4. clip the dataset to the min and max values
    im_contrast_adjusted = np.clip(im_data, min, max)

    return im_contrast_adjusted


def save_image(mrc_filename, output_file, BATCH_MODE, binning_factor, PRINT_SCALEBAR, scalebar_angstroms, angpix):
    check_dependencies()
    # need to recast imported module as the general keyword to use
    import PIL.Image as Image

    mrc_data = get_mrc_data(mrc_filename)

    print("mrc_filename = %s" % mrc_filename)

    ## apply sigma contrast to the image
    im_contrast_adjusted = apply_sigma_contrast(mrc_data, 3.5)

    ## rescale the image data to grayscale range (0,255)
    remapped = (255*(im_contrast_adjusted - np.min(im_contrast_adjusted))/np.ptp(im_contrast_adjusted)).astype(np.uint8) ## remap data from 0 -- 255

    ## load the image data into a PIL.Image object
    im = Image.fromarray(remapped).convert('RGB')

    ## figure out the name of the file
    if BATCH_MODE:
        ## in batch mode, inherit the base name of the .MRC file, just change the extension to the one provided by the user
        # output_format = os.path.splitext(output_file)[1].lower()
        output_format = os.path.splitext(sys.argv[1])[1]
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
        print("  >> image written: %s" % img_name )

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


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def check_dependencies():
    ## load built-in packages, if they fail to load then python install is completely wrong!
    globals()['sys'] = __import__('sys')
    globals()['os'] = __import__('os')
    globals()['glob'] = __import__('glob')
    globals()['mp'] = __import__('multiprocessing') ## similar to: import numpy as np

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

    import os
    import sys
    import glob
    import numpy as np
    from multiprocessing import Process, Pool
    import time

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
    ##################################

    ##################################
    ## ASSIGN DEFAULT VARIABLES
    ##################################
    GLOBAL_VARS = {
        'mrc_file' : str(),
        'output_file' : str(),
        'BATCH_MODE' : False,
        'BIN_IMAGE' : False,
        'binning_factor' : 4,
        'PRINT_SCALEBAR' : False,
        'scalebar_angstroms' : 200, # Angstroms
        'angpix' : 1.94,
        'threads' : 4
        }
    ##################################

    ##################################
    ## SET UP EXPECTED DATA FOR PARSER
    ##################################
    EXPECTED_FLAGS = {
     ##    flag      :  (GLOBAL_VARS_key,       data_type,  legal_entries/range,    is_toggle,  has_defaults)
        '--bin'      :  ('binning_factor'   ,    int(),     (1, 999),               False,      True ),
        '--scalebar' :  ('scalebar_angstroms',   int(),     (1, 9999),              False,      True ),
        '--angpix'   :  ('angpix',               float(),   (0.0001, 99999.999),    False,      True ),
        '--j'        :  ('threads',              int(),     (0,999),                False,      True)
    }

    EXPECTED_FILES = [  # cmd_line_index,   expected_extension,                         GLOBAL_VARS_key
                    (   1,                  '.mrc',                                     'mrc_file'),
                    (   2,                  ['.jpg', '.jpeg', '.png', '.tif', '.gif'],  'output_file')
                    ]
    ##################################

    start_time = time.time()

    parse_cmd_line(min_input = 1)

    ## add a custom checks outside scope of general parser above
    commands = []
    ## get all commands used
    for n in range(len(sys.argv[1:])+1):
        commands.append(sys.argv[n])
    ## check if --bin was given as a command, in which case toggle on the flag
    if '--bin' in commands:
        GLOBAL_VARS['BIN_IMAGE'] = True
    ## check if --scalebar was given as a command, in which case toggle on the flag
    if '--scalebar' in commands:
        GLOBAL_VARS['PRINT_SCALEBAR'] = True
    if not '--j' in commands:
        GLOBAL_VARS['threads'] = None

    ## print warning if no --angpix is given but --scalebar is (i.e. user may want to use a different pixel size)
    if GLOBAL_VARS['PRINT_SCALEBAR']:
        commands = []
        ## get all commands used
        for n in range(len(sys.argv[1:])+1):
            commands.append(sys.argv[n])
        ## check if --angpix was given
        if not '--angpix' in commands:
            print("!! WARNING: --scalebar was given without an explicit --angpix, using default value of 1.94 Ang/px !!")

    if not GLOBAL_VARS['BATCH_MODE']:
        ## single image conversion mode
        save_image(GLOBAL_VARS['mrc_file'], GLOBAL_VARS['output_file'], GLOBAL_VARS['BATCH_MODE'], GLOBAL_VARS['binning_factor'], GLOBAL_VARS['PRINT_SCALEBAR'], GLOBAL_VARS['scalebar_angstroms'], GLOBAL_VARS['angpix'])
    else:
        if GLOBAL_VARS['threads'] != None:
            ## permit multithreading
            threads = GLOBAL_VARS['threads']
            print(" ... multithreading activated (%s threads) " % threads)

            ## multithreading set up
            tasks = []
            for file in glob.glob("*.mrc"):
                tasks.append(file) ## inputs to the target function

            # task_indicies = np.arange(len(tasks)).tolist() ## will serve as our job id call table
            #
            #
            # ## prepare all processes by loading the desired arguments
            # processes = []
            # for task in tasks:
            #     processes.append(Process(target=save_image, args=(task, GLOBAL_VARS['output_file'], GLOBAL_VARS['BATCH_MODE'], GLOBAL_VARS['binning_factor'], GLOBAL_VARS['PRINT_SCALEBAR'], GLOBAL_VARS['scalebar_angstroms'], GLOBAL_VARS['angpix']), daemon = True))
            #
            # ## split the workload into logical chunks
            # batches = list(chunks(task_indicies, threads))
            #
            # ## dynamically run on multiple threads
            # for batch in batches:
            #     print(" >> Starting batch:")
            #     ## dynamically launch processes
            #     for job in batch:
            #         processes[job].start()
            #     ## dynamically have all launched processes join the main thread so it waits before proceeding
            #     for job in batch:
            #         processes[job].join(15)

            ### update 2021-09-02, try using Pool, which appears to be faster than above manual batching code
            try:
                ## define total workset inputs
                dataset = []
                for task in tasks:
                    dataset.append((task, GLOBAL_VARS['output_file'], GLOBAL_VARS['BATCH_MODE'], GLOBAL_VARS['binning_factor'], GLOBAL_VARS['PRINT_SCALEBAR'], GLOBAL_VARS['scalebar_angstroms'], GLOBAL_VARS['angpix']))
                ## prepare pool of workers
                pool = Pool(threads)
                ## assign workload to pool
                results = pool.starmap(save_image, dataset)
                ## close the pool from recieving any other tasks
                pool.close()
                ## merge with the main thread, stopping any further processing until workers are complete
                pool.join()

            except KeyboardInterrupt:
                print("Multiprocessing run killed")
                pool.terminate()


        else:
            ## get all files with extension
            for file in glob.glob("*.mrc"):
                GLOBAL_VARS['mrc_file'] = file
                save_image(GLOBAL_VARS['mrc_file'], GLOBAL_VARS['output_file'], GLOBAL_VARS['BATCH_MODE'], GLOBAL_VARS['binning_factor'], GLOBAL_VARS['PRINT_SCALEBAR'], GLOBAL_VARS['scalebar_angstroms'], GLOBAL_VARS['angpix'])

    end_time = time.time()
    total_time_taken = end_time - start_time
    print("Total time taken to run = %.2f sec" % total_time_taken)
    ## non-parallelized = 17.08 sec, 16.49 sec, 16.62 sec
    ## parallized, manual mode = 12.43 sec, 12.95 sec, 12.12 sec
    ## parallized, pool mode = 8.42 sec, 8.24 sec, 8.2 sec
