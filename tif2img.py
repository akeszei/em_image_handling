#!/usr/bin/env python3

"""
    A script to convert .TIF images to .JPG (or other) format with optional binning
    and scalebar.
"""

## 2024-09-19: Adapted from mrc2img.py to work for tif files 

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
    print(" Convert .TIF 2D EM images to conventional image formats (.PNG, .TIF, .GIF).")
    print(" Options include binning and addition of a scalebar of specified size.")
    print(" Usage:")
    print("    $ tif2img.py  input.tif  output.jpg/png/tif/gif  <options> ")
    print(" Batch mode:")
    print("    $ tif2img.py  /path/to/tif/@.jpg (or png,tif,gif)")
    print(" -----------------------------------------------------------------------------------------------")
    print(" Options (default in brackets): ")
    print("           --bin (4) : binning factor for image")
    print("    --scalebar (200) : add scalebar in Angstroms")
    print("       --angpix (-1) : Angstroms per pixel in .mrc image")
    print("   --batch_out (dir) : Choose a different path to save output files when using batch mode")
    print("             --j (4) : Allow multiprocessing using indicated number of cores")
    print("===================================================================================================")
    sys.exit()
    return

def get_xml_tag(xml_tree, header):
    """
    PARAMETERS 
        xml_tree = xml.etree.ElementTree.Element object 
        header = str(); case insensitive string of the header we want to capture (e.g. 'beamtilt')
    RETURNS
        str(); the element tag we can use to search directly for the children values of that header 
    """
    for elem in xml_tree.iter():
        raw_elem = remove_namespace(elem.tag)
        if header.lower() == raw_elem.lower():
            # print(" >> ", elem.tag)
            return elem.tag
    print(" !! ERROR :: Could not find the header (%s) in the XML file, doublecheck it exists in the file (note: case insensitive)" % header)
    exit()

def remove_namespace(input_string = ''):
    """ By default convension, the names of XML headers for EPU files contains a lot of unnecessary
        information, this function aims to remove all that unnecessary text and return the header name
        alone. e.g.:
            {namespace}headerTitle {} -> headerTitle
    """
    if '{' in input_string:
        # print( "input str = ", input_string)
        output_sting = input_string.split('{')[1].split('}')[1]
        return output_sting
    else: 
        return input_string

def get_pixel_size(xml_tree):
    search_header = 'pixelsize'
    ## get the namespaced version of the expected tag (case insensitive)
    search_string = get_xml_tag(xml_tree, search_header)
    ## enable recursion with './/' leader attached to the search string 
    matches = xml_tree.findall('.//' + search_string)
    if len(matches) == 0:
        print(" !! ERROR :: Could not find the given header in the XML file: %s (note: search is case insensitive)" % search_header)
        return 
    angpix = None 
    ## take first (and likely only) match retrieved by findall 
    for attrib in matches[0]:
        if 'x' in remove_namespace(attrib.tag).lower():
            ## cast the first 5 characters as a float (i.e. convert e-10 to angstroms)
            angstroms_per_pixel = float(attrib.text[:5]) 
            ## round to the nearest 2 decimal places 
            angstroms_per_pixel_rounded = np.round(angstroms_per_pixel, 2) 
            ## set the common variable 
            angpix = angstroms_per_pixel_rounded
    if angpix == None:
        print(" !! ERROR :: Could not find the attributes for the given header: %s, confirm there is an attribute for this header in the file!" % search_string)
        return 
    return angpix

def get_tif_data(file):
    """ file = .tif file name & path 
        returns np.ndarray of the mrc data using mrcfile module
    """

    try:
        with tifffile.TiffFile(file) as tif:
            print(" Opening tif file: %s" % file)
            for page in tif.pages:
                for tag in page.tags:
                    # print("         %s :: %s" % (tag.name, tag.value))
                    if tag.name == 'FEI_TITAN':
                        ## pass the XML data from the FEI_TITAN tag to a parser to grab the pixel value for x 
                        xml_tree = ET.fromstring(tag.value)
                        pixel_size = get_pixel_size(xml_tree)
                        print(" ... pixel size from header =", pixel_size)
    except:
        print(" There was a prblem reading the header for the input .TIF file (%s)" % file)
        exit()

    try:
        image_data = tifffile.imread(file)
    except:
        print(" There was a problem reading the .TIF file as an image ")
        exit()

    return image_data, pixel_size

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
    min = im_mean - (sigma_value * im_stdev)
    max = im_mean + (sigma_value * im_stdev)
    ## 4. clip the dataset to the min and max values
    im_contrast_adjusted = np.clip(im_data, min, max)

    return im_contrast_adjusted

def save_image(image_filename, output_file, BATCH_MODE, BIN_IMAGE, binning_factor, PRINT_SCALEBAR, scalebar_angstroms, input_angpix):
    check_dependencies()
    # need to recast imported module as the general keyword to use
    import PIL.Image as Image

    ## check file exists in the cwd
    if not os.path.isfile('./' + image_filename):
        print(" ERROR :: File (%s) does not exist in working directory!" % image_filename)
        sys.exit()

    ## read data from mrc file 
    image_data, image_pixel_size = get_tif_data(image_filename)

    ## check if a logical angpix value was given (default is -1) and use that instead, otherwise use value read from mrc file
    if input_angpix > 0:
        angpix = input_angpix
    else:
        angpix = image_pixel_size
        print(" Unexpected pixel size given (%s), detected pixel size from file used (%s)" % (input_angpix, angpix))

    ## if after above code runs pixel size does not yet make sense then set an arbitrary default and turn off functions that require angpix 
    if angpix <= 0 and PRINT_SCALEBAR:
        print(" Unexpected pixel size after parsing. Cannot print scalebar despite flag given!")
        PRINT_SCALEBAR = False

    ## apply sigma contrast to the image
    im_contrast_adjusted = apply_sigma_contrast(image_data, 3)

    ## rescale the image data to grayscale range (0,255)
    remapped = (255*(im_contrast_adjusted - np.min(im_contrast_adjusted))/np.ptp(im_contrast_adjusted)).astype(np.uint8) ## remap data from 0 -- 255

    ## load the image data into a PIL.Image object
    im = Image.fromarray(remapped).convert('RGB')

    ## figure out the name of the file
    if BATCH_MODE:
        ## in batch mode, inherit the base name of the .MRC file, just change the extension to the one provided by the user
        # output_format = os.path.splitext(output_file)[1].lower()
        output_format = os.path.splitext(sys.argv[1])[1]
        img_name = os.path.splitext(image_filename)[0] + output_format
    else:
        img_name = output_file

    if BIN_IMAGE:
        ## bin the image to the desired size
        try:
            resized_im = im.resize((int(im.width/binning_factor), int(im.height/binning_factor)), Image.Resampling.BILINEAR)
        except:
            print(" ERROR :: Old version of Pillow installed, try:")
            print("   pip install --upgrade Pillow")
            print(" ... and re-run code.")
            exit()
    else:
        ## if no --bin flag is given, set the binning factor to 1 instead of the default 4
        resized_im = im
        binning_factor = 1

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
    print(" Scale bar info: (length px, offset px, stroke) = (%s, %s, %s)" % (scalebar_px, indent_px, stroke))
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

    import os
    import sys
    import glob
    import numpy as np
    from multiprocessing import Process, Pool
    import time
    import cmdline_parser
    import tifffile ## python -m tifffile file.tif -> quickly load and look at the file 

    ## import element tree for XML parsing
    import xml.etree.ElementTree as ET 


    try:
        from PIL import Image
    except:
        print(" Could not import PIL.Image. Install depenency via:")
        print(" > pip install --upgrade Pillow")
        sys.exit()

    ##################################

    ##################################
    ## ASSIGN DEFAULT VARIABLES
    ##################################
    PARAMS = {
        'image_file' : str(),
        'output_file' : str(),
        'BATCH_MODE' : False,
        'batch_output_dir' : '.',
        'BIN_IMAGE' : False,
        'binning_factor' : 4,
        'PRINT_SCALEBAR' : False,
        'scalebar_angstroms' : 200, # Angstroms
        'angpix' : -1,
        'PARALLEL_PROCESSING': False,
        'threads' : 4
        }
    ##################################

    ##################################
    ## SET UP EXPECTED DATA FOR PARSER
    ##################################
    FLAGS = {
##    flag      :  (PARAMS_key,       data_type,  legal_entries/range,    toggle for entry,   intrinsic toggle,                    has_defaults)
    '--bin'      :  ('binning_factor'   ,    int(),     (1, 999),               False,              (True, 'BIN_IMAGE', True),             True ),
    '--scalebar' :  ('scalebar_angstroms',   int(),     (1, 9999),              False,              (True, 'PRINT_SCALEBAR', True),        True ),
    '--angpix'   :  ('angpix',               float(),   (0.0001, 99999.999),    False,              False,                                 True ),
    '--batch_out':  ('batch_output_dir',     str(),     (),                False,              False,   True),
    '--j'        :  ('threads',              int(),     (0,999),                False,              (True, 'PARALLEL_PROCESSING', True),   True)
    }


    FILES = { ## cmd line index    allowed extensions                                   ## can launch batch mode
        'image_file' : (      1,              '.tif',          False),
        'output_file' : (   2,              ['.jpg', '.jpeg', '.png', '.tif', '.gif'],  True)
        }
    ##################################

    start_time = time.time()
    PARAMS, EXIT_CODE = cmdline_parser.parse(sys.argv, 1, PARAMS, FLAGS, FILES)
    if EXIT_CODE < 0:
        usage()
        sys.exit()
    cmdline_parser.print_parameters(PARAMS, sys.argv)

    ## add a custom checks outside scope of general parser above
    commands = []
    ## get all commands used
    for n in range(len(sys.argv[1:])+1):
        commands.append(sys.argv[n])
    ## check if batch mode was enabled while also providing an explicit input file
    if PARAMS['BATCH_MODE'] and commands[1][-4:] == ".tif":
        print(" ERROR :: Cannot launch batch mode using `@' symbol while also providing an explicit input file!")
        print(" Remove the input file and re-run to enable batch mode processing")
        sys.exit()

    ## print warning if no --angpix is given but --scalebar is (i.e. user may want to use a different pixel size)
    if PARAMS['PRINT_SCALEBAR']:
        commands = []
        ## get all commands used
        for n in range(len(sys.argv[1:])+1):
            commands.append(sys.argv[n])
        ## check if --angpix was given
        if not '--angpix' in commands:
            print("!! WARNING: --scalebar was given without an explicit --angpix, will try to parse from file instead !!")

    if not PARAMS['BATCH_MODE']:
        ## warn the user if they activated parallel processing for a single image
        if '--j' in commands:
            print(" NOTE: --j flag was set for parallel processing, but without batch mode. Only 1 core can be used for processing a single image.")
        ## single image conversion mode
        save_image(PARAMS['image_file'], PARAMS['output_file'], PARAMS['BATCH_MODE'], PARAMS['BIN_IMAGE'], PARAMS['binning_factor'], PARAMS['PRINT_SCALEBAR'], PARAMS['scalebar_angstroms'], PARAMS['angpix'])
    else:
        if PARAMS['PARALLEL_PROCESSING']:
            ## permit multithreading
            threads = PARAMS['threads']
            print(" ... multithreading activated (%s threads) " % threads)

            ## multithreading set up
            tasks = []
            for file in glob.glob("*.tif"):
                tasks.append(file) ## inputs to the target function

            ### update 2021-09-02, try using Pool, which appears to be faster than above manual batching code 
            try:
                ## define total workset inputs
                dataset = []
                for task in tasks:
                    dataset.append((task, PARAMS['output_file'], PARAMS['BATCH_MODE'], PARAMS['BIN_IMAGE'], PARAMS['binning_factor'], PARAMS['PRINT_SCALEBAR'], PARAMS['scalebar_angstroms'], PARAMS['angpix']))
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
            for file in glob.glob("*.tif"):
                PARAMS['image_file'] = file
                save_image(PARAMS['image_file'], PARAMS['output_file'], PARAMS['BATCH_MODE'], PARAMS['BIN_IMAGE'], PARAMS['binning_factor'], PARAMS['PRINT_SCALEBAR'], PARAMS['scalebar_angstroms'], PARAMS['angpix'])

    end_time = time.time()
    total_time_taken = end_time - start_time
    print("... runtime = %.2f sec" % total_time_taken)
    ## non-parallelized = 17.08 sec, 16.49 sec, 16.62 sec
    ## parallized, manual mode = 12.43 sec, 12.95 sec, 12.12 sec
    ## parallized, pool mode = 8.42 sec, 8.24 sec, 8.2 sec
