#!/usr/bin/env python3

"""
    A script to stack .MRC images to one larger .MRCS file
"""

## 2021-10-26: Script started
## 2023-03-21: Polished script to automatically handle header data & mrc modes. Allow user input to specify output file name, choose a different directory than the current one, and use a reference file to chose which .MRC files to sort and in what order 
## 2023-04-05: Fixed bug to remove path from files when retrieved by glob 

#############################
###     FLAGS/GLOBALS
#############################
DEBUG = False
PRINT_EXECUTION_TIME = False
MRC_MODES = {   0  : 'int8', ## see: https://mrcfile.readthedocs.io/en/stable/usage_guide.html#data-types
                1  : 'int16',
                2  : 'float32', 
                4  : 'complex64', 
                6  : 'uint16', 
                12 : 'float16' }

#############################
###     DEFINITION BLOCK
#############################

def usage():
    print("===================================================================================================")
    print(" Convert a set of .MRC files in working directory into one stacked .MRCS file")
    print(" Usage:")
    print("    $ stack_mrc.py  ")
    print("---------------------------------------------------------------------------------------------------")
    print(" Options: ")
    print("               --dir (.) : Read .MRC files from a target directory other than the working directory")
    print("      --o (stacked.mrcs) : Change the name of the output .MRCS file. Can specify relative path.")
    print("           --sort (file) : By default, images are sorted alphanumerically before stacking, however") 
    print("                           a text file can be used to indicate which files to stack, and in what ")
    print("                           order. E.g.:")
    print("                               First_image_99.mrc")
    print("                               Some_other_image_003.mrc")
    print("                               Final_image_in_stack_01.mrc")
    ## maybe add --mode (2) : bypass automated detection of .MRC mode and use this one instead?
    print("---------------------------------------------------------------------------------------------------")
    print(" Example with options:")
    print("    $ stack_mrc.py --dir path/to/mrc/ --o processed/stack.mrcs --sort path/to/mrc/sort.txt ")
    print("===================================================================================================")
    sys.exit()
    return

def parse_cmdline(cmdline):
    if DEBUG:
        print("=====================================================")
        print(" PARSING COMMAND LINE:")
        print("-----------------------------------------------------")

    ## check for a help flag and exit early if found aywhere 
    for cmd in cmdline:
        if cmd in ['--h', '--H', '--help']:
            usage()

    ## setup defaults 
    dir = "."
    output_mrcs_fname = 'stacked.mrcs'
    sort_list_fname = ''

    ## setup flags 
    EXPECTED_FLAGS = ['--dir', '--o', '--sort']
    ## parse each entry on the command line
    for i in range(len(cmdline)):
        cmd = cmdline[i]        

        ## look only for expected flags and present a warning if the parse finds something unusual
        if i > 0:
            ## flags only appear in odd indices: 0 *1* 2 *3* ...
            if i%2 == 1:
                if cmd in EXPECTED_FLAGS:
                    pass
                else: 
                    print(" !! ERROR :: Unexpected command line formatting (%s is not a flag)" % cmd)
                    usage()
            else:
                ## skip even entries as they cannot be flags 
                continue 

        if cmd == "--dir":
            if len(cmdline) > i and cmdline[i + 1] not in EXPECTED_FLAGS:
                dir = cmdline[i + 1].rstrip("/\\") ## remove trailing slashes for consistency
            
            ## check the directory actually exists 
            if not os.path.isdir(dir):
                print(" !! ERROR :: directory given by the --dir flag could not be found (%s)" % dir)

        if cmd == "--o":
            ## find if there exists another commmand
            if len(cmdline) > i and cmdline[i + 1] not in EXPECTED_FLAGS:
                output_mrcs_fname = cmdline[i + 1]
            else:
                print(" !! WARNING :: No input was provided for flag: %s" % cmd)
        
        if cmd == "--sort":
            ## find if there exists another commmand
            if len(cmdline) > i  and cmdline[i + 1] not in EXPECTED_FLAGS:
                sort_list_fname = cmdline[i + 1]
            else:
                print(" !! WARNING :: No input file was given for flag: %s" % cmd)
                print("    ... using default alphanumeric sorting")
            ## check the file pointed to exists 
            if not os.path.isfile(sort_list_fname):
                print(" !! ERROR :: file given by the --sort flag could not be found (%s)" % sort_list_fname)
                exit()

    if DEBUG: 
        print(" >> Reading .MRC files from directory = ", dir)
        if sort_list_fname != '':
            print(" >> Stacking .MRC files according to = ", sort_list_fname)
        print(" >> Output filename = ", output_mrcs_fname)
        print("=====================================================")

    return dir, output_mrcs_fname, sort_list_fname

def get_mrc_filenames(dir, sort_files_list = ''):
    """ Find all .mrc files in the working directory, or those specified by a given list in a text file  
    """
    ## check the input dir has a leading slash
    if dir[-1] != '/':
        ## if it does not, then add one
        dir = dir + "/"
    
    files = []

    if sort_files_list == '':
        ## if no input file was provided, use glob matching and sort function to prepare the list 
        for file in glob.glob(dir + "*.mrc"):
            files.append(os.path.basename(file)) ## inputs to the target function
        ## make sure list is sorted by alpha numeric
        files = sorted(files)
    else:
        ## create the list of files based on the input file 
        with open(sort_files_list) as f:
            for line in f:
                line = line.strip() ## remove wrapping whitespace
                ## check the expected file exists before adding it 
                if os.path.isfile(dir + line):
                    files.append(line)
                else:
                    print(" !! ERROR :: File (%s) indicated in sort list (%s) not found, exiting" % (dir + line, sort_files_list))
                    exit()


    if DEBUG:
        print(" ... %s .MRC files found" % len(files))

    return files

def get_mrc_image(file):
    """ file = .mrc file
        returns np.ndarray of the mrc data using mrcfile module
    """
    with mrcfile.open(file) as mrc:
        image_data = mrc.data
    return image_data

def make_empty_mrcs(stack_size, mrc_dimensions, mrc_mode, fname, apix):
    """ Prepare an empty .MRCS in memory of the correct dimensionality
    PARAMETERS 
        stack_size = int(), how many images will be stacked into this .MRCS 
        mrc_dimensions = list( int(), int() ), size of individual .MRC files to be stacked in the form: [ x, y ]
        mrc_mode = int(), mode of the mrc (necessary to set proper data type)
        fname = str(), full absolution or relative path of the output .MRCS file 
        apix = float(), the voxel size to set the mrcs file to 
    """

    if mrc_mode in [0, 1, 2, 4, 6, 12]:
        ## we have a compatible mode detected
        if DEBUG:
            print(" ... using MRC mode #%s -> dtype = %s " % (mrc_mode, MRC_MODES[int(mrc_mode)])) 
    else:
        ## we do not have a compatible mode! 
        print("!! ERROR :: Unknown mode read from input file, %s -> mode %s" % (fname, mrc_mode))
        exit()

    with mrcfile.new(fname, overwrite=True) as mrc:
        mrc.set_data(np.zeros(( stack_size, ## stack size
                                mrc_dimensions[1], ## pixel height, 'Y'
                                mrc_dimensions[0]  ## pixel length, 'X'
                            ), dtype=np.dtype(MRC_MODES[int(mrc_mode)])))
                            # ), dtype=np.float32))

        ## set pixel size data 
        mrc.voxel_size = apix
        mrc.update_header_from_data()
        mrc.update_header_stats()
        ## set the mrcfile with the correct header values to indicate it is an image stack
        mrc.set_image_stack()
        if DEBUG:
            print(" ... writing new .MRCS stack (%s, mode %s) with dimensions (x, y, z) = (%s, %s, %s)" % (fname, mrc_mode, mrc.data.shape[2], mrc.data.shape[1], mrc.data.shape[0]))
    return

def get_mrc_header(dir, fname):
    """
        Get the necessary header information from an mrc file 
    PARAMETERS 
        dir = str(), absolute or relative path of the folder containing the mrc files in question (e.g. /data/mrc/)
        fname = str(), full name of the mrc file to read from (e.g. Image_0001.mrc)
    RETURNS 
        x_dim = int(), pixel size of the image along the x-axis 
        y_dim = int(), pixel size of the image along the y-axis 
        mrc_mode = int(), mode of the .MRC file (see mrcfile documentation) 
        apix = float(), A/pix in the header 
    """
    if DEBUG: 
        print(" get_mrc_header :: ")
        print("   >> dir = %s" % dir)
        print("   >> fname = %s" % fname)

    ## check the input dir has a leading slash
    if dir[-1] != '/':
        ## if it does not, then add one
        dir = dir + "/"

    with mrcfile.open(dir + fname, mode='r') as mrc:
        y_dim, x_dim = mrc.data.shape[0], mrc.data.shape[1]
        mrc_mode = mrc.header.mode
        apix = mrc.voxel_size.x

    # X axis is always the last item in shape (see: https://mrcfile.readthedocs.io/en/latest/usage_guide.html)
    if DEBUG:
        print(" ... input .MRC dimensions (x, y) = (%s, %s), MRC mode = %s, angpix = %s" % (x_dim, y_dim, mrc_mode, apix))
    return x_dim, y_dim, mrc_mode, apix

def write_mrc_to_stack(dir, mrcs_name, stack_index, input_mrc_name):
    """
    PARAMETERS
        dir = str(); the target directory to look for individual .MRC files to load
        mrcs_name = str(); file name of the .MRCS
        stack_index = int(); position in the .MRCS stack to write/slice into
        input_mrc_name = str(); file name with the .MRC image data to read from
    """
    ## check the input dir has a leading slash
    if dir[-1] != '/':
        ## if it does not, then add one
        dir = dir + "/"

    print(" ... writing: %s to stack at index %s" % (os.path.basename(input_mrc_name), stack_index))
    with mrcfile.open(dir + input_mrc_name, mode='r') as mrc:
        mrc_data = mrc.data

    with mrcfile.open(mrcs_name, mode='r+') as mrcs:
        mrcs.data[stack_index] = mrc_data
        mrcs.update_header_from_data()
        mrcs.update_header_stats()

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

    print("=====================================================")
    print(" RUNNING :: stack_mrc.py ")
    print("-----------------------------------------------------")

    ## set up default variables and parse input 
    dir, output_mrcs_fname, sort_list_fname = parse_cmdline(sys.argv)

    input_mrc_files = get_mrc_filenames(dir, sort_list_fname)

    start_time = time.time()

    mrc_dimensions = [0, 0]
    mrc_dimensions[0], mrc_dimensions[1], mrc_mode, apix = get_mrc_header(dir, input_mrc_files[0])

    output_mrcs = make_empty_mrcs(len(input_mrc_files), mrc_dimensions, mrc_mode, output_mrcs_fname, apix)

    for n in range(len(input_mrc_files)):
        current_frame = input_mrc_files[n]
        write_mrc_to_stack(dir, output_mrcs_fname, n, current_frame)

    print("-----------------------------------------------------")
    print(" COMPLETE :: file written >> %s " % output_mrcs_fname)
    print("=====================================================")

    end_time = time.time()
    total_time_taken = end_time - start_time
    if PRINT_EXECUTION_TIME:
        print("... runtime = %.2f sec" % total_time_taken)
