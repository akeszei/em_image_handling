#!/usr/bin/env python3

from dataclasses import dataclass

## Written by: Alexander Keszei
## 2022-05-01: mrcs_viewer.py version 1 complete
## 2023-04-27: Updated to correctly display .mrcs files of raw exposures (i.e. tilt series). Also added other functionality.
## 2026-09-30: Adapted from mrcs_viewer.py to make mrcs_slices.py

"""
To Do:
    - Set options for giving specific slices to make instead of just 10% of dataset 
"""

@dataclass
class Parameters:
    in_fpath: str = "" # mrc/mrcs file whose header we want to copy from
    out_png_fpath: str = "" # location and name of png we want to write out 
    scaling_factor: float = 0.5

    def usage(self):
        print("================================================================================================================")
        print(" Read mrc stack and write out integrated slices along z axis representing 10% of the data each step.")
        print(" Usage:")
        print("    $ mrcs_slices.py  /path/to/ref.mrc  /path/to/save.png  ")
        print(" -----------------------------------------------------------------------------------------------")
        print(" Options (default in brackets): ")
        print("             --scale (0.5) : each integrated slice by this factor")
        print("================================================================================================================")
        sys.exit()
        return 

    def parse_cmdline(self):
        cmdline = sys.argv

        ## cmd line minimally needs 3 inputs
        if len(cmdline) < 2:
            self.usage()

        ## first check for the help flag before proceeding 
        for i in range(len(cmdline)):
            if cmdline[i] in ['-h', '--h', '-H', '--H']:
                self.usage()
        
        ## iterate over every entry, looking for flags & files  
        for i in range(len(cmdline)):
            cmd = cmdline[i]

            ## look for a .mrc file 
            if len(cmd) > len('.mrc'):
                if cmd[-len('.mrc'):].lower() == '.mrc':
                    self.assign_mrcfile(cmd)
            ## also try for .mrcs file 
            if len(cmd) > len('.mrcs'):
                if cmd[-len('.mrcs'):].lower() == '.mrcs':
                    self.assign_mrcfile(cmd)

            ## look for output file location 
            if len(cmd) > len('.png'):
                if cmd[-len('.png'):].lower() == '.png':
                    self.set_pngfile(cmd)

        ## Deal with flags 
        for i in range(len(cmdline)):
            cmd = cmdline[i]

            if cmd == '--scale':

                try:
                    self.scaling_factor = float(cmdline[i + 1])
                except:
                    print(" ERROR :: Could not --scale flag  ")

        return 

    def assign_mrcfile(self, f):
        ## first instance of an mrc file will be the reference
        if self.in_fpath == '':
            ## check the input file exists 
            if os.path.exists(f):
                self.in_fpath = f
                # print("    Reference mrc to read header from: ", self.in_fpath)
            else:
                print(" !! ERROR :: Input file (%s) does not exist! Check path carefully " % f)
                exit()
        else:
            print(" ERROR !! More than two .mrc files were detected on the cmdline, check your inputs. ")
            self.usage()
        return 

    def set_pngfile(self, f):
        if self.out_png_fpath == '':
            ## check the input file exists 
            if os.path.exists(os.path.dirname(f)  or './'):
                self.out_png_fpath = f
                # print("    Write out png file at: ", self.out_png_fpath)
            else:
                print(" !! ERROR :: Path given (%s) does not exist! " % f)
                exit()
        else:
            print(" ERROR !! More than two .png outputs were specified on the cmdline, check your inputs. ")
            self.usage()

        return 

    def print_parameters(self):
        print(" Parameters:")
        print("----------------------------------------")
        print("  input mrc file : ", self.in_fpath)
        print(" output png file : ", self.out_png_fpath)
        print("  scaling factor : ", self.scaling_factor)
        return 

#region Global functions

def get_mrcs_raw_data(mrcs_file_path):
    with mrcfile.open(mrcs_file_path) as mrcs:
        d = mrcs.data.astype(np.float32)
    return d

def get_normalized_mrcs_slice(mrcs_file_path, x, y, scaling_factor = 0.1):

    ## open the mrcs file as an nparray of dimension (z, box_size, box_size), where z is the number of images in the stack
    with mrcfile.open(mrcs_file_path) as mrcs:
        print(" Get slice from mrc file: %s -> %s (%s total slices in stack)" % (x, y, mrcs.data.shape[0]))
        
        ## deal with single frame .mrcs files as a special case
        if len(mrcs.data.shape) == 2:
            print(" ERROR :: MRCS file has only one image")
            sys.exit()

        ## throw error if slice limit is greater than what exists in the stack 
        if mrcs.data.shape[0] < y:
            print(" ERROR :: Requested slice from %s -> %s, but there are only %s image in the file stack!" % (x, y, mrcs.data.shape[0]))
            sys.exit()

        counter = -1
        ## interate over the mrcs stack and grab the slices we want
        for n in range(mrcs.data.shape[0]):
            counter +=1
            
            if counter == x:
                print(" !!! first counter = ", counter)
                ## create the first image 
                integrated_slice = mrcs.data[n].astype(np.float32)
            elif counter > x and counter < y:
                print(" ??? counter = ", counter)
                current_img = mrcs.data[n].astype(np.float32)
                integrated_slice += current_img

        normalized_slice = get_grayscale_img(integrated_slice, scaling_factor)


    return normalized_slice


def mrc2grayscale(mrc_raw_data, VERBOSE = False):
    """ Convert raw mrc data into a grayscale numpy array suitable for display
    """
    ## remap the mrc data to grayscale range
    remapped = (255*(mrc_raw_data - np.min(mrc_raw_data))/np.ptp(mrc_raw_data)).astype(np.uint8) ## remap data from 0 -- 255 as integers
    return remapped

def sigma_contrast(im_array, sigma, VERBOSE = False):
    """ Rescale the image intensity levels to a range defined by a sigma value (the # of
        standard deviations to keep). 
    """
    stdev = np.std(im_array)
    mean = np.mean(im_array)
    minval = mean - (stdev * sigma)
    if minval < 0:
        minval = 0
    maxval = mean + (stdev * sigma)
    if maxval > 255:
        maxval = 255

    if VERBOSE:
        print("======================================")
        print(" sigma_contrast (s = %s)" % sigma)
        print("--------------------------------------")
        print("  stdev = %s" % stdev)
        print("  mean = %s" % mean)
        print("  input min, max = (%s, %s)" % (np.min(im_array), np.max(im_array)))
        print("  cutoff min, max = (%s, %s)" % (minval, maxval))

    ## remove pixles above/below the defined limits
    im_array = np.clip(im_array, minval, maxval)
    ## rescale the image into the range 0 - 255
    im_array = ((im_array - minval) / (maxval - minval)) * 255

    return im_array.astype('uint8')

def get_mrcs_info(fname, VERBOSE = False):
    """ Retrieve image dimensions, stack size and relevant header info from the file 
    """
    with mrcfile.open(fname, mode='r') as mrc:
        ## deal with single frame mrcs files as special case
        if len(mrc.data.shape) == 2:
            y_dim, x_dim = mrc.data.shape[0], mrc.data.shape[1]
            z_dim = 1
        else:
            ## X axis is always the last in shape (see: https://mrcfile.readthedocs.io/en/latest/usage_guide.html)
            y_dim, x_dim, z_dim = mrc.data.shape[1], mrc.data.shape[2], mrc.data.shape[0]

        ## Read pixel size        
        pixel_size = mrc.voxel_size['x']  
        ## Read the dtype of the image array 
        dtype = mrc.data.dtype

    if VERBOSE:
        print("======================================")
        print(" get_mrcs_info (%s) " % fname)
        print("--------------------------------------")
        print("  (x, y, z) = (%s, %s, %s)" % (x_dim, y_dim, z_dim) )
        print("  pixel_size = %s" % pixel_size)
        print("  dtype = %s " % dtype)

    return x_dim, y_dim, z_dim, pixel_size, dtype


def resize_image(im_array, scaling_factor, VERBOSE = False):
    ## calculate the new dimensions based on the scaling factor and input image
    scaled_width = int(im_array.shape[1] * scaling_factor)
    scaled_height = int(im_array.shape[0] * scaling_factor)

    if VERBOSE:
        print("======================================")
        print(" resize_image (scaling factor = %s)" % scaling_factor)
        print("--------------------------------------")
        print("  %s -> (%s, %s) " % (im_array.shape, scaled_width, scaled_height))

    # resized_im = cv2.resize(im_array, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST) ## for int arrays use INTER_NEAREST
    resized_im = cv2.resize(im_array, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA) ## for noisy micrographs, default INTER_LINEAR does not work well, switch to INTER_AREA

    return resized_im

def get_grayscale_img(raw_im, scaling_factor, VERBOSE = False):
    remapped = mrc2grayscale(raw_im, VERBOSE = VERBOSE)
    remapped = sigma_contrast(remapped, 4, VERBOSE = VERBOSE)
    scaled = resize_image(remapped, scaling_factor, VERBOSE = VERBOSE)

    return scaled 


def create_image_array(imgs, ncols = 5, padding = 2, order_to_print = None, VERBOSE = False):
    """
    PARAMETERS 
        imgs = list of image array data, e.g.: [ np.array(img1), ..., ]
        ncols = int(); defining how many images should populate each row 
        padding = int(); pixels distance between images 
        order_to_print = list() of integers, indicating a specific order to display the images (i.e. [3, 2, 1, 0] would be reverse order); numbers should not repeat and the list length should match the input array of images! 
    """

    PADDING = padding

    ## calculate the number of rows based on the number total images and the desired column number 
    if len(imgs) <= ncols:
        nrows = 1
    else:
        nrows = math.ceil(len(imgs) / ncols) 

    ## get the image size (should be a perfect square so only grab one dimension)
    image_box_size = imgs[0].shape[0]

    if VERBOSE:
        print(" Prepare image array ::")
        print("   # images = %s" % len(imgs))
        print("   img box size = %s" % image_box_size)
        print("   img array type & shape = ", type(imgs[0]), imgs[0].shape)
        print("   array dimensios to print = %s x %s (col x rows)" % (ncols, nrows))

    image_format = '.png'
    ## prepare a blank canvas to draw upon, if .PNG format add empty alpha channel
    if image_format.lower() == ".png":
        canvas = np.full((nrows * image_box_size + PADDING * (nrows - 1), ncols * image_box_size + PADDING * (ncols - 1), 2), (0, 0), np.uint8) ## by convention, alpha is last channel
    else:
        canvas = np.full((nrows * image_box_size + PADDING * (nrows - 1), ncols * image_box_size + PADDING * (ncols - 1)), np.inf) ## by convention, top left of the image is coordinate (0, 0)

    ## populate the canvas with each image at a specific location
    counter = 0
    if ncols > len(imgs):
        num_imgs_to_print = len(imgs)
    else:
        num_imgs_to_print = nrows * ncols
    for n in range(num_imgs_to_print):
        col = counter % ncols
        row = int(counter / ncols)
        if VERBOSE:
            print("panel position (col, row) = (%s, %s)" % (col, row))

        y_range = (row * image_box_size + (PADDING * row), (row * image_box_size) + image_box_size + (PADDING * row))
        x_range = (col * image_box_size + (PADDING * col), (col * image_box_size) + image_box_size + (PADDING * col))

        if VERBOSE:
            print("x and y ranges = ", x_range, y_range)

        ## check if we have an image at this index
        if len(imgs) - 1 >= counter:
            ## 'stamp' the image onto the target location
            if image_format.lower() == ".png":
                ## add alpha channel data to the incoming image
                alpha = np.full((image_box_size, image_box_size), 255, np.uint8) ## remove full transparency from area where image will be displayed
                image_RGBA = np.dstack((imgs[n], alpha)) ## apply the new transparency values to the image area in question
                canvas[ y_range[0]: y_range[1] , x_range[0] : x_range[1]] = image_RGBA

            else:
                canvas[ y_range[0]: y_range[1] , x_range[0] : x_range[1]] = imgs[n]

        counter += 1

    return canvas

def add_scalebar(im, box_size, angpix, scalebar_size, indent_px = 8, stroke = 4, VERBOSE = False):
    if angpix == 0:
        print(" ERROR :: Pixel size not known (currently detected as 0). Will not add scalebar.")
        return im

    scalebar_px = int(scalebar_size / angpix)
    if scalebar_px > box_size:
        print(" ERROR : Requested scalebar size (%s Ang, %s px) exceeds the dimensions of the image (%s px)!" % (scalebar_size, scalebar_px, box_size))
        sys.exit()

    ## find the pixel range for the scalebar, typically 5 x 5 pixels up from bottom left
    LEFT_INDENT = indent_px # px from left to indent the scalebar
    BOTTOM_INDENT = indent_px # px from bottom to indent the scalebar
    STROKE = stroke # px thickness of scalebar
    x_range = (LEFT_INDENT, LEFT_INDENT + scalebar_px)
    y_range = (box_size - BOTTOM_INDENT - STROKE, box_size - BOTTOM_INDENT)

    ## set the pixels white for the scalebar
    for x in range(x_range[0], x_range[1]):
        for y in range(y_range[0], y_range[1]):
            im[y][x] = 255

    if VERBOSE:
        print(" Printing scalebar onto first panel:")
        print("   >> %s pixels (%s Angstroms)" % (scalebar_px, scalebar_size))
        print("   >> Indent scalebar %s pixels from bottom left edge of panel" % indent_px)
        print("-------------------------------------------------------------")

    return im

#endregion 


##########################
#region RUN BLOCK
##########################
if __name__ == '__main__':
    import numpy as np
    import os, string, sys, math
    from PIL import Image as PIL_Image
    import mrcfile
    import cv2 ## for resizing images with a scaling factor


    params = Parameters()
    params.parse_cmdline()

    params.print_parameters()

    ## set the values here
    f = params.in_fpath
    scaling_factor = params.scaling_factor
    save_path= params.out_png_fpath

    x_dim, y_dim, z_dim, pixel_size, dtype = get_mrcs_info(f)

    ## how many slices represent 10% of the z height
    ten_percent_chunk = z_dim // 10 

    ## generate a list of 10% steps we can use to call the slice generator 
    slices = []
    for i in range(0, z_dim, ten_percent_chunk):
        x = i
        y = i + ten_percent_chunk
        slices.append((x,y))
        
    ## load mrcs data in memory 
    raw_data = get_mrcs_raw_data(f)

    ## read the mrcs data in memory and pull desired slices from it, sum them, the normalize and convert to grayscale 
    imgs = []
    for (x,y) in slices:
        data_slice = raw_data[x:y, :, :] 
        integrated_slice = np.sum(data_slice, axis = 0)
        normalized_slice = get_grayscale_img(integrated_slice, scaling_factor)
        imgs.append(normalized_slice)


    im_array = create_image_array(imgs)
    img = PIL_Image.fromarray(im_array)
    img.save(save_path)
    print(" Written file: ", save_path)
    # img.show()

#endregion 