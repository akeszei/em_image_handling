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


def get_ser_data(file):
    import serReader
    ## use serReader module to parse the .SER file data into memory
    im = serReader.serReader(file)
    ## get the image data as an np.ndarray of dimension x, y from TIA .SER image
    im_data = im['data']
    ## grab useful metadata
    pixelSize = "{0:.4g}".format(im['pixelSizeX'] * 10**10)
    ## recast the int32 data coming out of serReader into float32 for use as mrc mode #2
    im_float32 = im_data.astype(np.float32)
    return im_float32, pixelSize

def save_mrc_image(im_data, output_name, pixelSize):
    """
        im_data = needs to be np.array float32 with dimensions of the TIA image
        output_name = string; name of the output file to be saved
        pixelSize = voxel size taken from the ser file
    """
    with mrcfile.new(output_name, overwrite = True) as mrc:
        mrc.set_data(im_data)
        mrc.voxel_size = pixelSize
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
        ## adjust the contrast
        remapped = sigma_contrast(remapped, 3)
        ## load the image data into a PIL.Image object
        im = Image.fromarray(remapped).convert('L')
        ## bin the image to the desired size
        resized_im = im.resize((int(im.width/binning_factor), int(im.height/binning_factor)), Image.Resampling.BILINEAR)
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
    # print("Image intensity data (mean, stdev) = (%s, %s)" % (mean, stdev))
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
    im_data, pixelSize = get_ser_data(ser_file)
    ## save the data to a .MRC file
    save_mrc_image(im_data, mrc_file, pixelSize)
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
    if int(Image.__version__.split('.')[0]) < 9:
        print(" This script requires Pillow version 9+")
        print(" ... detected version = %s" % Image.__version__)
        print(" Try upgrading vis:")
        print("  > pip install pillow --upgrade")

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
