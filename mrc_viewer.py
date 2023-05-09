#!/usr/bin/env python3

## Author: Alexander Keszei
## 2022-05-11: Version 1 finished
## 2023-03-27: Updated to improve filtering on fast implementation (switching interpolation mode on resize function was crucial)
"""
To Do:
    - Somehow lock the left/right keys from firing while loading an image. Basic attempts to do this by adding a flag on the start/end of the load_img function fails since the keystrokes are queued and fire after the function completes 
    - Add ctf toggle with 'c' keyboard press 
""" 

DEBUG = False
start_time = None

def usage():
    print("====================================================================================")
    print("   A simple Tk-based .MRC viewer with useful single-particle image analysis tools. ")
    print("------------------------------------------------------------------------------------")
    print("   - Left/Right arrow keys iterate over .MRC files in directory/")
    print("   - Left click with 'Show particle picks' on to draw circles of a given size.")
    print("   - Adjust scale, filter (0 = no filter), and sigma contrast using right-hand menus")
    print("   - Use 'Apply Scaled first' for speed, turn off for full resolution filtering (slow).")
    print("   - Use the File > Open menu to choose a specific file, or write its full name")
    print("        in the input text widget at the top.")
    print("   - Save out displayed image with File > Save menu.")
    print("====================================================================================")
    return

def speedtest(method):
    global start_time
    if method == 'start': 
        start_time = time.time()
    
    if method == 'stop':
        end_time = time.time()
        total_time_taken = end_time - start_time
        print("... runtime = %.2f sec" % total_time_taken)
    return

def add_scalebar(image_obj, scalebar_px, scalebar_stroke):
    """ Adds a scalebar to the input image and returns a new edited image
    PARAMETERS
        image_obj = PIL.Image object
    """
    ## set the indentation to be ~2.5% inset from the bottom left corner of the image
    indent_px = int(image_obj.height * 0.025)
    # ## set the stroke to be ~0.5% image size
    # scalebar_stroke = int(image_obj.height * 0.005)
    # if scalebar_stroke < 1:
    #     scalebar_stroke = 1

    if DEBUG: print("Scale bar info: (offset px, stroke) = (%s, %s)" % (indent_px, scalebar_stroke))
    ## find the pixel range for the scalebar, typically 5 x 5 pixels up from bottom left
    LEFT_INDENT = indent_px # px from left to indent the scalebar
    BOTTOM_INDENT = indent_px # px from bottom to indent the scalebar
    STROKE = scalebar_stroke # px thickness of scalebar
    x_range = (LEFT_INDENT, LEFT_INDENT + scalebar_px)
    y_range = (image_obj.height - BOTTOM_INDENT - STROKE, image_obj.height - BOTTOM_INDENT)

    ## set the pixels white for the scalebar
    for x in range(x_range[0], x_range[1]):
        for y in range(y_range[0], y_range[1]):
            # image_obj.putpixel((x, y), (255, 255, 255))
            image_obj.putpixel((x, y), (255)) # grayscale so 1 dimension
    return image_obj

def find_file_index(fname, directory_list, IGNORE_CAPS = True):
    """ For a given file name and list of files in directory, return the index that corresponds to the input file name or None if not present
    """
    fname = os.path.basename(fname)
    i = 0
    for file_w_path in directory_list:
        basename = os.path.basename(file_w_path)
        if fname.lower() == basename.lower():
            return i
        i += 1

    ## if we have not returned an index, then return an error
    if DEBUG: print(" No match found for file (%s)" % fname)
    return None

def sigma_contrast(im_array, sigma):
    """ Rescale the image intensity levels to a range defined by a sigma value (the # of
        standard deviations to keep). Can perform better than auto_contrast when there is
        a lot of dark pixels throwing off the level balancing.
    """
    import numpy as np
    stdev = np.std(im_array)
    mean = np.mean(im_array)
    minval = mean - (stdev * sigma)
    maxval = mean + (stdev * sigma)

    if minval < 0: 
        minval = 0
    if maxval > 255:
        maxval = 255

    if DEBUG:
        print(" sigma_contrast (s = %s)" % sigma)

    ## remove pixels above/below the defined limits
    im_array = np.clip(im_array, minval, maxval)
    ## rescale the image into the range 0 - 255
    im_array = ((im_array - minval) / (maxval - minval)) * 255

    return im_array.astype('uint8')

def gamma_contrast(im_array, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    ## REF: https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    im = cv2.LUT(im_array, table)
    if DEBUG: print(" gamma_contrast (g = %s)" % gamma)
    return im

def lowpass2(img, threshold, pixel_size):
    """ Another example of a fast implementation of a lowpass filter (not used here)
        ref: https://wsthub.medium.com/python-computer-vision-tutorials-image-fourier-transform-part-3-e65d10be4492
    """

    ## create circle mask at a resolution given by the threshold and pixel size
    radius = int(img.shape[0] * pixel_size / threshold)
    if DEBUG: print(" FFT mask radius calculated for %s ang (%s apix) is %s" % (threshold, pixel_size, radius))
    mask = np.zeros_like(img)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx,cy), radius, (255,255), -1)[0]
    ## blur the mask
    lowpass_mask = cv2.GaussianBlur(mask, (19,19), 0)

    f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = lowpass_mask * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.abs(inv_img)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img*255 / filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)

    ## to view the fft we need to play with the results
    f_abs = np.abs(f_complex)
    f_bounded = 20 * np.log(f_abs) # we take the logarithm of the absolute value of f_complex, because f_abs has tremendously wide range.
    f_img = 255 * f_bounded / np.max(f_bounded) ## convert data to grayscale based on the new range
    f_img = f_img.astype(np.uint8)
    return filtered_img, f_img

def lowpass(img, threshold, pixel_size):
    if DEBUG:
        print(" Low pass filter image by %s Ang (%s angpx)" % (threshold, pixel_size))

    ## generate 2D FFT as complex output
    im_fft_raw = np.fft.fft2(img)
    ## apply shift of origin to center of image
    im_fft = np.fft.fftshift(im_fft_raw)

    ## for display purposes, adjust the fft image to a suitable range 
    im_fft_abs = np.abs(im_fft) # combine real & imaginary components   
    im_fft_bounded = 20 * np.log(im_fft_abs) # the natural FT has a large dynamic range, compress it and rescale it for better contrast
    im_fft_display = 255 * im_fft_bounded / np.max(im_fft_bounded) # fit the signal into the grayscale image range (0 - 255)
    im_fft_display = im_fft_display.astype(np.uint8) # recast as a uint8 array which is the expected format for a grayscale image by more programs

    ## if we have a zero or negative value for the lowpass threshold, it means we are not applying a filter
    ## just return the input image and calculated CTF
    if threshold <= 0:
        return img, im_fft_display

    ## prepare the lowpass filter by creating a circular mask at a resolution given by the threshold and pixel size
    radius = int(img.shape[0] * pixel_size / threshold) # in Fourier space, this is the distance from the central origin that corresponds to the given resolution threshold (assuming the image is square)
    if DEBUG: print(" FFT mask radius calculated for %s ang is %s" % (threshold, radius))
    mask = np.zeros_like(img) # prepare a black canvas 
    cy = mask.shape[0] // 2 # find the central y-axis coordinate of the image
    cx = mask.shape[1] // 2 # find the central x-axis coordinate of the image
    cv2.circle(mask, (cx,cy), radius, (255,255), -1) # set all pixels of a centered circle of given radius to white 
    # cv2.circle(mask, (cx,cy), radius, (255,255), -1)[0] #
    gauss_kernel = (9, 9) # must be odd ## NOTE: An arbitrary kernel size is used here, maybe adjust dynamically based on image size?
    soft_mask = cv2.GaussianBlur(mask, gauss_kernel, 0) # blur the mask to avoid hard edges in Fourier space

    # ## Uncomment to display the mask
    # cv2.imshow('', soft_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ## convolve the lowpass mask with the centered image FFT to apply filter
    filtered_fft = np.multiply(im_fft,soft_mask) / 255

    ## reverse the steps to regenerate the filtered image 
    filtered_fft_backshift = np.fft.ifftshift(filtered_fft) # shift origin back from center to upper left corner
    filtered_im_complex = np.fft.ifft2(filtered_fft_backshift) # calculate inverse FFT
    filtered_im = np.abs(filtered_im_complex).clip(0,255).astype(np.uint8) # combine complex real and imaginary components and clip it back into a working range for a grayscale image

    return filtered_im, im_fft_display

def resize_image(img_nparray, scaling_factor):
    """ Uses OpenCV to resize an input grayscale image (0-255, 2d array) based on a given scaling factor
            scaling_factor = float()
    """
    original_width = img_nparray.shape[1]
    original_height = img_nparray.shape[0]
    scaled_width = int(img_nparray.shape[1] * scaling_factor)
    scaled_height = int(img_nparray.shape[0] * scaling_factor)
    if DEBUG: print("resize_img function, original img_dimensions = ", img_nparray.shape, ", new dims = ", scaled_width, scaled_height)
    resized_im = cv2.resize(img_nparray, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA) 
    # resized_im = cv2.resize(img_nparray, (scaled_width, scaled_height)) ## note: default interpolation is INTER_LINEAR, and does not work well for noisy EM micrographs 
    return resized_im

def get_mrc_files_in_dir(path):
    files_in_dir = os.listdir(path)
    images = []
    for file in files_in_dir:
        extension = file[-4:]
        if extension.lower() in [".mrc"]:
            images.append(os.path.join(path, file))
    return sorted(images)

def get_fixed_array_index(current_object_number, maximum_number_of_columns):
    """ For a given fixed array width (column max) and object number, return its corresponding array index
    """
    row = int(current_object_number / maximum_number_of_columns)
    col = current_object_number % maximum_number_of_columns
    # print(" get_fixed_array_index -> (%s, %s)" % (row, col))
    return row, col

def get_PhotoImage_obj(img_nparray, ADD_SCALEBAR = False, scalebar_px = 50, scalebar_stroke = 5):
    """ Convert an input numpy array grayscale image and return an ImageTk.PhotoImage object
    """
    PIL_img = PIL_Image.fromarray(img_nparray.astype(np.uint8))  #.convert('L')

    if ADD_SCALEBAR:
        PIL_img = add_scalebar(PIL_img, scalebar_px, scalebar_stroke)

    img_obj = ImageTk.PhotoImage(PIL_img)
    return img_obj

def get_mrc_raw_data(file):
    """ file = .mrc file
        returns np.ndarray of the mrc data using mrcfile module
    """
    ## NOTE atm works for mrc mode 2 but not mode 6
    with mrcfile.open(file) as mrc:
        image_data = mrc.data.astype(np.float32) ## need to cast it as a float32, since some mrc formats are uint16! (i.e. mode 6)
        # print("mrc dtype = ", type(image_data.dtype))
        pixel_size = np.around(mrc.voxel_size.item(0)[0], decimals = 2)
    if DEBUG:
        print(" Opening %s" % file)
        print("   >> image dimensions (x, y) = (%s, %s)" % (image_data.shape[1], image_data.shape[0]))
        print("   >> pixel size = %s Ang/px" % pixel_size)

    ## set defaults 
    if pixel_size == 0:
        pixel_size = 1.0

    return image_data, pixel_size

def mrc2grayscale(mrc_raw_data, pixel_size, lowpass_threshold):
    """ Convert raw mrc data into a grayscale numpy array suitable for display
    """
    # print(" min, max = %s, %s" % (np.min(mrc_raw_data), np.max(mrc_raw_data)))
    ## remap the mrc data to grayscale range
    remapped = (255*(mrc_raw_data - np.min(mrc_raw_data))/np.ptp(mrc_raw_data)).astype(np.uint8) ## remap data from 0 -- 255 as integers

    # lowpassed, ctf = lowpass(remapped, lowpass_threshold, pixel_size) # ~0.8 sec
    lowpassed, ctf = lowpass2(remapped, lowpass_threshold, pixel_size) # ~0.7 sec

    return lowpassed, ctf

def coord2freq(x, y, fft_width, fft_height, angpix):
    """ For a given coordinate in an FFT image, return the frequency spacing corresponding to that position (i.e. resolution ring)
    PARAMETERS
        x = int(), x-axis pixel position in centered FFT
        y = int(), y-axis pixel position in centered FFT
        fft_width = int(), magnitude (in pixels) of x-axis
        fft_height = int(), magnitude (in pixels) of y-axis
        angpix = float(), resolution corresponding to real space image
    RETURNS
        frequency = float(), resolution (in Angstroms) corresponding to that pixel position
        difference_vector_magnitude = float(), the length of the ring corresponding to that frequency spacing (mainly needed for drawing)
    """
    ## calculate the angular distance of the picked coordinate from the center of the image
    FFT_image_center_coordinate = ( int(fft_width / 2), int(fft_height / 2))
    difference_vector = tuple(b - a for a, b in zip( (x, y), FFT_image_center_coordinate))
    difference_vector_magnitude = np.linalg.norm(difference_vector)
    ## convert the magnitude value into frequency value in units angstroms
    frequency = ( 1 / ( difference_vector_magnitude / fft_width ) ) * angpix

    if DEBUG:
        print(" ==============================================================================")
        print(" coord2freq :: inputs:")
        print("   pixel position = (%s, %s)" % (x, y))
        print("   im dimensions = (%s, %s)" % (fft_width, fft_height))
        print("   im angpix = %.2f" % angpix)
        print("  >> FFT image center coordinate = (%s, %s)" % FFT_image_center_coordinate)
        print(" >> Vector magnitude in pixels = ", difference_vector_magnitude)
        print(" >> Frequency of radial pixel position = " + "{:.2f}".format(frequency) + " Ang")
        print(" ==============================================================================")
    return frequency, difference_vector_magnitude

def gaussian_blur(self, im_array, sigma, DEBUG = True):
    if DEBUG:
        print("=======================================")
        print(" image_handler :: gaussian_blur")
        print("---------------------------------------")
        print("  input img dim = ", im_array.shape)
        print("  sigma = ", sigma)
        print("=======================================")

    blurred_img = ndimage.gaussian_filter(im_array, sigma)
    return blurred_img

def auto_contrast(self, im_array, DEBUG = True):
    """ Rescale the image intensity levels to a reasonable range using the top/bottom 2 percent
        of the data to define the intensity levels
    """
    ## avoid hotspot pixels by looking at a group of pixels at the extreme ends of the image
    minval = np.percentile(im_array, 2)
    maxval = np.percentile(im_array, 98)

    if DEBUG:
        print("=======================================")
        print(" image_handler :: auto_contrast")
        print("---------------------------------------")
        print("  input img dim = ", im_array.shape)
        print("  original img min, max = (%s, %s)" % (np.min(im_array), np.max(im_array)))
        print("  stretch to new min, max = (%s %s)" % (minval, maxval))
        print("=======================================")

    ## remove pixles above/below the defined limits
    im_array = np.clip(im_array, minval, maxval)
    ## rescale the image into the range 0 - 255
    im_array = ((im_array - minval) / (maxval - minval)) * 255

    return im_array


class MainUI:
    def __init__(self, instance, start_index):
        self.instance = instance
        instance.title("Tk-based MRC viewer")
        # instance.geometry("520x500")
        # instance.resizable(False, False)

        ## CLASS VARIABLES
        self.displayed_widgets = list() ## container for all widgets packed into the main display UI, use this list to update each
        self.display_data = list() ## container for the image objects for each displayed widgets (they must be in the scope to be drawn)
                                   ## in this program, display_data[0] will contain the scaled .jpg-style image; while display_data[1] will contain the display-ready CTF for that image
        self.display_im_arrays = list() ## contains the nparray versions of the image/ctf currently in the display window (for saving jpgs) 
        self.mrc_dimensions = ('x', 'y')
        self.pixel_size = float()
        self.image_name = str()
        self.scale_factor = 0.25 ## scaling factor for displayed image
        self.lowpass_threshold = 14
        self.sigma_contrast = 3
        self.SHOW_PICKS = tk.BooleanVar(instance, True)
        self.picks_diameter = 150 ## Angstroms, `picks' are clicked particles by the user
        self.picks_color = 'red'
        self.coordinates = dict() ## list of picked points
        self.SHOW_SCALEBAR = tk.BooleanVar(instance, False)
        self.scalebar_length = 200 ## Angstroms
        self.scalebar_stroke = 5 ## pixels
        self.scalebar_color = 'white'
        self.SHOW_CTF = tk.BooleanVar(instance, False)
        self.index = start_index ## 0 ## index of the list of known mrc files int he directory to view
        self.working_dir = "."
        self.SPEED_OVER_ACCURACY = tk.BooleanVar(instance, True)

        ## MENU BAR LAYOUT
        self.initialize_menubar(instance)

        ## MAIN WINDOW WITH SCROLLBARS
        self.image_name_label = image_name_label = tk.Entry(instance, font=("Helvetica", 16), highlightcolor="blue", borderwidth=None, relief=tk.FLAT, foreground="black", background="light gray")
        # image_name_label.pack(fill='both', padx= 20)
        image_name_label.grid(row = 0 , column =  0, sticky = (tk.EW), padx = 5)
        self.viewport_frame = viewport_frame =  ttk.Frame(instance)

        right_side_panel_fontsize = 10

        ## MRC INFO
        self.iminfo_header = tk.Label(instance, font=("Helvetica, 16"), text="MRC info")
        self.iminfo_header.grid(row = 1, column = 1, columnspan = 2) #, sticky = (tk.N, tk.W))

        self.MRC_dimensions_LABEL = tk.Label(instance, font=("Helvetica", right_side_panel_fontsize), text="(%s, %s)" % ('x', 'y'))
        self.MRC_dimensions_LABEL.grid(row = 2, column = 1, columnspan = 2)

        self.MRC_angpix_LABEL = tk.Label(instance, font=("Helvetica", right_side_panel_fontsize), text="%s Å/px" % '1')
        self.MRC_angpix_LABEL.grid(row = 3, column = 1, columnspan = 2)

        self.MRC_displayed_angpix_LABEL = tk.Label(instance, font=("Helvetica", right_side_panel_fontsize), text="Display @ %s Å/px" % '2')
        self.MRC_displayed_angpix_LABEL.grid(row = 4, column = 1, columnspan = 2)


        ## DISPLAY SETTINGS
        self.separator = ttk.Separator(instance, orient='horizontal')
        self.separator.grid(row=5, column =1, columnspan = 2, sticky=tk.EW)
        self.settings_header = tk.Label(instance, font=("Helvetica, 16"), text="Display")
        self.settings_header.grid(row = 6, column = 1, columnspan = 2) #, sticky = (tk.N, tk.W))

        self.scale_LABEL = tk.Label(instance, font=("Helvetica", right_side_panel_fontsize), text="Scale factor: ")
        self.scale_ENTRY = tk.Entry(instance, width=4, font=("Helvetica", right_side_panel_fontsize))
        self.scale_LABEL.grid(row = 7, column = 1, sticky = (tk.N, tk.E))
        self.scale_ENTRY.grid(row = 7, column = 2, sticky = (tk.N, tk.W))

        self.lowpass_threshold_LABEL = tk.Label(instance, font=("Helvetica", right_side_panel_fontsize), text="Lowpass: ")
        self.lowpass_threshold_ENTRY = tk.Entry(instance, width=4, font=("Helvetica", right_side_panel_fontsize))
        self.lowpass_threshold_LABEL.grid(row = 8, column = 1, sticky = (tk.N, tk.E))
        self.lowpass_threshold_ENTRY.grid(row = 8, column = 2, sticky = (tk.N, tk.W))

        self.sigma_contrast_LABEL = tk.Label(instance, font=("Helvetica", right_side_panel_fontsize), text="Sigma: ")
        self.sigma_contrast_ENTRY = tk.Entry(instance, width=4, font=("Helvetica", right_side_panel_fontsize))
        self.sigma_contrast_LABEL.grid(row = 9, column = 1, sticky = (tk.N, tk.E))
        self.sigma_contrast_ENTRY.grid(row = 9, column = 2, sticky = (tk.N, tk.W))

        self.show_CTF_TOGGLE = tk.Checkbutton(instance, text='Display CTF', variable=self.SHOW_CTF, onvalue=True, offvalue=False, command=self.toggle_SHOW_CTF)
        self.show_CTF_TOGGLE.grid(row = 10, column = 1, columnspan = 2, sticky = (tk.N, tk.W))

        self.speed_over_accuracy_TOGGLE = tk.Checkbutton(instance, text='Apply scale first', variable=self.SPEED_OVER_ACCURACY, onvalue=True, offvalue=False, command = self.toggle_SPEED_OVER_ACCURACY)
        self.speed_over_accuracy_TOGGLE.grid(row = 11, column = 1, columnspan = 2, sticky = (tk.N, tk.W))


        ## OPTIONAL SETTINGS
        self.separator2 = ttk.Separator(instance, orient='horizontal')
        self.separator2.grid(row=12, column =1, columnspan = 2, sticky=tk.EW)
        self.optional_header = tk.Label(instance, font=("Helvetica, 16"), text="Optional")
        self.optional_header.grid(row = 13, column = 1, columnspan = 2) #, sticky = (tk.N, tk.W))

        self.show_picks_TOGGLE = tk.Checkbutton(instance, text='Show particle picks', variable=self.SHOW_PICKS, onvalue=True, offvalue=False, command=self.toggle_SHOW_PICKS)
        self.show_picks_TOGGLE.grid(row = 14, column = 1, columnspan = 2, sticky = (tk.N, tk.W))

        self.picks_diameter_LABEL = tk.Label(instance, font=("Helvetica", right_side_panel_fontsize), text="Diameter (Å): ")
        self.picks_diameter_ENTRY = tk.Entry(instance, width=4, font=("Helvetica", right_side_panel_fontsize))
        self.picks_diameter_LABEL.grid(row = 15, column = 1, sticky = (tk.N, tk.E))
        self.picks_diameter_ENTRY.grid(row = 15, column = 2, sticky = (tk.N, tk.W))

        self.show_scalebar_TOGGLE = tk.Checkbutton(instance, text='Display scalebar', variable=self.SHOW_SCALEBAR, onvalue=True, offvalue=False, command=self.toggle_SHOW_SCALEBAR)
        self.show_scalebar_TOGGLE.grid(row = 16, column = 1, columnspan = 2, sticky = (tk.N, tk.W))

        self.scalebar_length_LABEL = tk.Label(instance, font=("Helvetica", right_side_panel_fontsize), text="Scalebar (Å): ")
        self.scalebar_length_ENTRY = tk.Entry(instance, width=4, font=("Helvetica", right_side_panel_fontsize))
        self.scalebar_length_LABEL.grid(row = 17, column = 1, sticky = (tk.N, tk.E))
        self.scalebar_length_ENTRY.grid(row = 17, column = 2, sticky = (tk.N, tk.W))

        viewport_frame.grid(row = 1, column = 0, rowspan = 100)

        scrollable_frame, viewport_canvas = self.initialize_scrollable_window(self.viewport_frame)

        ## LOAD AN INITIAL MRC FILE
        self.next_img('none')

        ## SET THE SIZE OF THE PROGRAM WINDOW BASED ON THE SIZE OF THE DATA FRAME AND THE SCREEN RESOLUTION
        self.resize_program_to_fit_screen_or_data()

        ## EVENT BINDING
        instance.bind("<Configure>", self.resize) ## Bind manual screen size adjustment to updating the scrollable area

        ## KEYBINDINGS
        self.instance.bind("<F1>", lambda event: self.debugging())
        # self.instance.bind("<F2>", lambda event: self.redraw_canvases())
        # self.instance.bind('<Control-KeyRelease-s>', lambda event: self.save_selected_mrcs())

        self.instance.bind('<Left>', lambda event: self.next_img('left'))
        self.instance.bind('<Right>', lambda event: self.next_img('right'))
        self.instance.bind('<z>', lambda event: self.next_img('left'))
        self.instance.bind('<x>', lambda event: self.next_img('right'))
        # self.instance.bind('<c>', lambda event: self.toggle_SHOW_CTF()) ## will need to set the toggle

        self.image_name_label.bind('<Control-KeyRelease-a>', lambda event: self.select_all(self.image_name_label))
        self.image_name_label.bind('<Return>', lambda event: self.image_name_updated())
        self.image_name_label.bind('<KP_Enter>', lambda event: self.image_name_updated())
        self.scale_ENTRY.bind('<Control-KeyRelease-a>', lambda event: self.select_all(self.scale_ENTRY))
        self.scale_ENTRY.bind('<Return>', lambda event: self.scale_updated())
        self.scale_ENTRY.bind('<KP_Enter>', lambda event: self.scale_updated())
        self.lowpass_threshold_ENTRY.bind('<Control-KeyRelease-a>', lambda event: self.select_all(self.lowpass_threshold_ENTRY))
        self.lowpass_threshold_ENTRY.bind('<Return>', lambda event: self.lowpass_threshold_updated())
        self.lowpass_threshold_ENTRY.bind('<KP_Enter>', lambda event: self.lowpass_threshold_updated())
        self.sigma_contrast_ENTRY.bind('<Control-KeyRelease-a>', lambda event: self.select_all(self.sigma_contrast_ENTRY))
        self.sigma_contrast_ENTRY.bind('<Return>', lambda event: self.sigma_updated())
        self.sigma_contrast_ENTRY.bind('<KP_Enter>', lambda event: self.sigma_updated())
        self.picks_diameter_ENTRY.bind('<Control-KeyRelease-a>', lambda event: self.select_all(self.picks_diameter_ENTRY))
        self.picks_diameter_ENTRY.bind('<Return>', lambda event: self.pick_diameter_updated())
        self.picks_diameter_ENTRY.bind('<KP_Enter>', lambda event: self.pick_diameter_updated())
        self.scalebar_length_ENTRY.bind('<Control-KeyRelease-a>', lambda event: self.select_all(self.scalebar_length_ENTRY))
        self.scalebar_length_ENTRY.bind('<Return>', lambda event: self.scalebar_updated())
        self.scalebar_length_ENTRY.bind('<KP_Enter>', lambda event: self.scalebar_updated())


        ## PANEL INSTANCES
        # self.optionPanel_instance = None
        return

    def save_jpg(self):
        suggested_jpg_fname = os.path.splitext(self.image_name)[0] + ".jpg"

        file_w_path = asksaveasfilename(  parent = self.instance, initialfile = suggested_jpg_fname,
                            defaultextension=".jpg",filetypes=[("All Files","*.*"),("JPEG","*.jpg")])

        save_dir, save_name = os.path.split(str(file_w_path))
        # print("File selected: ", file_name)
        # print("Working directory: ", file_dir)

        if self.SHOW_CTF.get() == True:
            img_to_save = self.display_im_arrays[1]
        else:
            img_to_save = self.display_im_arrays[0]
    
        cv2.imwrite(file_w_path, img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 100])

        ## open the image to draw coordinate picks if activated 
        if self.SHOW_PICKS.get() == True:
            img_to_write = cv2.imread(file_w_path)
            ## box_size is a value given in Angstroms, we need to convert it to pixels
            display_angpix = self.pixel_size / self.scale_factor
            box_width = self.picks_diameter / display_angpix
            box_halfwidth = int(box_width / 2)

            for coordinate in self.coordinates:
                # print("writing coordinate @ =", coordinate)
                cv2.circle(img_to_write, coordinate, box_halfwidth, (0,0,255), 2)
            
            cv2.imwrite(file_w_path, img_to_write, [cv2.IMWRITE_JPEG_QUALITY, 100])

        ## open the image to draw a scalebar if activated 
        if self.SHOW_SCALEBAR.get() == True:
            img_to_write = cv2.imread(file_w_path)
            scalebar_px = int(self.scalebar_length / (self.pixel_size / self.scale_factor))
            scalebar_stroke = self.scalebar_stroke
            indent_x = int(self.display_im_arrays[0].shape[0] * 0.025)
            indent_y = self.display_im_arrays[0].shape[1] - int(self.display_im_arrays[0].shape[1] * 0.025)
            
            cv2.line(img_to_write, (indent_x, indent_y), (indent_x + scalebar_px, indent_y), (255, 255, 255), scalebar_stroke)

            cv2.imwrite(file_w_path, img_to_write, [cv2.IMWRITE_JPEG_QUALITY, 100])

        print(" Saved display to >> %s" % file_w_path)

        return

    def rescale_picked_coordinates(self, old_scale, new_scale):
        """ Rescale coordinate positions to match a new scale.
        PARAMETERS
            self = instance of MainUI
            old_scale = original scaling factor for the image/coordinates
            new_scale = updated scaling factor for the image/coordinates
        """
        if DEBUG: print(" rescale particle positions from %s -> %s" % (old_scale, new_scale))
        new_coordinates = []
        ## rescale each coordinate by the necessary amount
        for coord in self.coordinates:
            new_coord = (int(coord[0] * new_scale / old_scale), int(coord[1] * new_scale / old_scale))
            if DEBUG: print("     x, y : %s --> %s" % (coord, new_coord))
            new_coordinates.append(new_coord)

        ## update the list of coordinates known to the instance
        self.coordinates = new_coordinates

        ## redraw coordinates
        self.draw_image_coordinates()
        return

    def is_clashing(self, mouse_position):
        """
        PARAMETERS
            mouse_position = tuple(x, y); pixel position of the mouse when the function is called
        """
        ## Calculate the boundaries to use for the clash check
        display_angpix = self.pixel_size / self.scale_factor
        box_width = self.picks_diameter / display_angpix
        box_halfwidth = int(box_width / 2)
        # print(" box width in pixels = %s" % box_width)

        for (x_coord, y_coord) in self.coordinates:
            if DEBUG:
                print(" CLASH TEST :: mouse_position = ", mouse_position, " ; existing coord = " , x_coord, y_coord)
            ## check x-position is in range for potential clash
            if x_coord - box_halfwidth <= mouse_position[0] <= x_coord + box_halfwidth:
                ## check y-position is in range for potential clash
                if y_coord - box_halfwidth <= mouse_position[1] <= y_coord + box_halfwidth:
                    ## if both x and y-positions are in range, we have a clash
                    del self.coordinates[(x_coord, y_coord)] # remove the coordinate that clashed
                    return True # for speed, do not check further coordinates (may have to click multiple times for severe overlaps)
        return False

    def draw_image_coordinates(self):
        """ Read a dictionary of pixel coordinates and draw boxes centered at each point
        """
        canvas = self.displayed_widgets[0]

        ## delete any pre-existing coordinates if already drawn
        canvas.delete('particle_positions')

        ## check if we are allowed to draw coordinates before proceeding
        if self.SHOW_PICKS.get() == False:
            return

        ## sanity check we are not looking at the CTF image
        if self.SHOW_CTF.get() == True:
            return

        ## box_size is a value given in Angstroms, we need to convert it to pixels
        display_angpix = self.pixel_size / self.scale_factor
        box_width = self.picks_diameter / display_angpix
        box_halfwidth = int(box_width / 2)

        for coordinate in self.coordinates:
            ## each coordinate is the center of a box, thus we need to offset by half the gif_box_width pixel length to get the bottom left and top right of the rectangle
            x0 = coordinate[0] - box_halfwidth
            y0 = coordinate[1] - box_halfwidth
            x1 = coordinate[0] + box_halfwidth
            y1 = coordinate[1] + box_halfwidth #y0 - img_box_size # invert direction of box to take into account x0,y0 are at bottom left, not top left
            # canvas.create_rectangle(x0, y0, x1, y1, outline='red', width=1, tags='particle_positions')
            canvas.create_oval(x0, y0, x1, y1, outline=self.picks_color, width=2, tags='particle_positions')
        if DEBUG: print(" %s image coordinates drawn onto canvas" % len(self.coordinates))
        return

    def draw_resolution_ring(self, pixel_coordinate):
        """ Let the user click on an FFT image and draw the corresponding resolution ring
        PARAMETERS
            self = instance of MainUI
            pixel_coordinate = tuple(int(), int()); coordinate of pixel position on the image in question (x, y)
        """
        canvas = self.displayed_widgets[0]
        ## delete any pre-existing resolution ring information
        canvas.delete('ctf_markup')

        ## get the frequency data from the pixel position
        x, y = pixel_coordinate
        displayed_angpix = self.pixel_size / self.scale_factor
        fft_width = canvas.winfo_width()
        fft_height = canvas.winfo_height()
        freq, diff_vector_magnitude = coord2freq(x, y, fft_width, fft_height, displayed_angpix)
        FFT_image_center_coordinate = (int(fft_width / 2), int(fft_height / 2))

        ## use simple logic to place the resolution text on the display in a visible location
        estimated_text_box_size = (60, 20) ## (x, y)
        if pixel_coordinate[0] + estimated_text_box_size[0] > fft_width - 8:
            text_box_x = pixel_coordinate[0] - estimated_text_box_size[0] - 5
        else:
            text_box_x = x
        if pixel_coordinate[1] - estimated_text_box_size[1] < 8 :
            text_box_y = pixel_coordinate[1] + estimated_text_box_size[1] + 8
        else:
            text_box_y = y

        if self.SHOW_CTF.get():
            canvas.create_text(text_box_x + 5, text_box_y - 4, font=("Helvetica", 14), text = "{:.2f}".format(freq) + " Å", fill='red', anchor = tk.SW,  tags='ctf_markup')

            ## draw a guiding line that shows the vector being measured from center of image to the mouse position
            canvas.create_line( FFT_image_center_coordinate, pixel_coordinate, fill='yellow', width=1, tags='ctf_markup') # line goes through the series of points (x0, y0), (x1, y1), … (xn, yn)

            ## draw a guiding circle to indicate the resolution ring being measured
            canvas.create_oval(FFT_image_center_coordinate[0] - int(diff_vector_magnitude), FFT_image_center_coordinate[1] - int(diff_vector_magnitude), FFT_image_center_coordinate[0] + int(diff_vector_magnitude), FFT_image_center_coordinate[1] + int(diff_vector_magnitude), dash = (7,4,2,4 ), width = 2, outline = 'red', tags='ctf_markup') # Creates a circle or an ellipse at the given coordinates. It takes two pairs of coordinates; the top left and bottom right corners of the bounding rectangle for the oval.

        return

    def on_left_mouse_down(self, x, y):
        """ Add coordinates to the dictionary at the position of the cursor, then call a redraw.
        """
        mouse_position = x, y

        if self.SHOW_CTF.get() == False:
            ## when clicking, check the mouse position against loaded coordinates to figure out if the user is removing a point or adding a point
            if self.is_clashing(mouse_position): # this function will also remove the point if True
                pass
            else:
                if DEBUG: print("Add coordinate: x, y =", mouse_position[0], mouse_position[1])
                x_coord = mouse_position[0]
                y_coord = mouse_position[1]
                self.coordinates[(x_coord, y_coord)] = 'new_point'
            self.draw_image_coordinates()
        else:
            self.draw_resolution_ring(mouse_position)
        return

    def load_file(self):
        """ Permits the system browser to be launched to select an image
            form a directory. Loads the directory and file into their
            respective variables and returns them
        """
        # See: https://stackoverflow.com/questions/9239514/filedialog-tkinter-and-opening-files
        file_w_path = askopenfilename(parent=self.instance, initialdir=".", title='Select file', filetypes=(
                                            # ("All files", "*.*"),
                                            ("Medical Research Council format", "*.mrc"),
                                            ))
        if file_w_path:
            try:
                # extract file information from selection
                file_dir, file_name = os.path.split(str(file_w_path))
                # print("File selected: ", file_name)
                # print("Working directory: ", file_dir)
                self.working_dir = file_dir
                self.load_img(file_w_path)

            except:
                showerror("Open Source File", "Failed to read file\n'%s'" % file_w_path)
            return

    def next_img(self, direction):
        """ Increments the current image index based on the direction given to the function.
        """
        ## Check if an entry widget has focus, in which case do not run this function
        # print(" %s has focus" % self.instance.focus_get())
        active_widget = self.instance.focus_get()
        if isinstance(active_widget, tk.Entry):
            if DEBUG: print(" Entry widget has focus, do not run next_img function")
            return

        ## find the files in the working directory
        image_list = get_mrc_files_in_dir(self.working_dir)

        ## adjust the index based on the input type
        if direction == 'right':
            self.index += 1
            # reset index to the first image when going past the last image in the list
            if self.index > len(image_list)-1 :
                self.index = 0
        if direction == 'left':
            self.index -= 1
            # reset index to the last image in the list when going past 0
            if self.index < 0:
                self.index = len(image_list)-1

        if DEBUG: print(" Load next image: %s" % image_list[self.index])
        self.load_img(image_list[self.index])

        # ## clear global variables for redraw
        # self.reset_globals()

        # ## load image with index 'n'
        # self.load_img()
        return

    def toggle_SHOW_CTF(self):
        """
        """
        if self.SHOW_CTF.get() == True:
            if DEBUG: print(" Display CTF")
            self.load_img_on_canvas(self.displayed_widgets[0], self.display_data[1])
        else:
            if DEBUG: print(" Display regular image")
            self.load_img_on_canvas(self.displayed_widgets[0], self.display_data[0])
            ## draw image coordinates if necessary
            self.draw_image_coordinates()

        return

    def toggle_SPEED_OVER_ACCURACY(self):
        """
        """
        if DEBUG: print(" Speed over accuracy toggle actuated, reload image.")
        ## remove the active canvas from the display 
        self.destroy_active_canvases()
        ## to temporarily display an empty canvas, we need to make a call to the gui to force a redraw
        self.scrollable_frame.update() ## update the frame holding the data
        ## start loading the next image with the new toggle setting 
        self.load_img(image_list[self.index])
        return

    def toggle_SHOW_PICKS(self):
        """
        """
        if self.SHOW_PICKS.get() == True:
            if DEBUG: print(" Display picked coordinates")
            ## WIP update the display window
        else:
            if DEBUG: print(" Hide picked coordinates")
            ## WIP reload the regular image
        ## draw image coordinates if necessary
        self.draw_image_coordinates()

        return

    def pick_diameter_updated(self):
        user_input = self.picks_diameter_ENTRY.get().strip()
        ## cast the input to an integer value
        try:
            user_input = int(user_input)
        except:
            self.picks_diameter_ENTRY.delete(0, tk.END)
            self.picks_diameter_ENTRY.insert(0,self.picks_diameter)
            print(" Input requires integer values > 0")
        ## check if input is in range
        if user_input >= 0:
            if DEBUG: print("particle pick diameter updated: %s" % user_input )
            self.picks_diameter = user_input
            ## pass focus back to the main instance
            self.instance.focus()

            self.next_img('none') ## WIP: redraw whole canvas probably not necessary... instead redraw only the markup!
        else:
            self.picks_diameter_ENTRY.delete(0, tk.END)
            self.picks_diameter_ENTRY.insert(0,self.picks_diameter)
            print(" Input requires positive integer values")
        return

    def toggle_SHOW_SCALEBAR(self):
        """
        """
        ## reload the image since the toggle has been changed
        self.next_img("none")
        return

    def image_name_updated(self):
        user_input = self.image_name_label.get().strip()

        ## find the files in the working directory
        image_list = get_mrc_files_in_dir(self.working_dir)

        if DEBUG: print(" Look for input fname (%s) in working dir (%s)" % (user_input, self.working_dir))
        match_index = find_file_index(user_input, image_list)
        if match_index == None:
            print(" No match found for image in working directory:")
            print("     >> Image = %s" % user_input)
            print("     >> Working dir = %s" % self.working_dir)
            self.image_name_label.delete(0, tk.END)
            self.image_name_label.insert(0," No match found")
            ## pass focus back to the main instance
            self.instance.focus()
        else:
            self.index = match_index
            ## pass focus back to the main instance
            self.instance.focus()
            self.next_img('none')
            ## reset the size of the main program
            self.resize_program_to_fit_screen_or_data()
        return

    def scale_updated(self):
        user_input = self.scale_ENTRY.get().strip()
        ## cast the input to a float value
        try:
            user_input = float(user_input)
        except:
            self.scale_ENTRY.delete(0, tk.END)
            self.scale_ENTRY.insert(0,self.scale_factor)
            print(" Input requires float values [10,0]")
        ## check if input is in range (prevent making it too large!)
        if 11 > user_input > 0:
            if DEBUG: print(" set scale factor to %s" % user_input )
            ## rescale any existing particle picks to approximately the same position
            self.rescale_picked_coordinates(self.scale_factor, user_input)
            ## update the scale factor on the instance
            self.scale_factor = user_input
            ## pass focus back to the main instance
            self.instance.focus()
            self.next_img('none')
            ## reset the size of the main program
            self.resize_program_to_fit_screen_or_data()

        else:
            self.scale_ENTRY.delete(0, tk.END)
            self.scale_ENTRY.insert(0,self.scale_factor)
            print(" Input requires float values [10,0]")
        return

    def lowpass_threshold_updated(self):
        user_input = self.lowpass_threshold_ENTRY.get().strip()
        ## cast the input to an integer value
        try:
            user_input = int(user_input)
        except:
            self.lowpass_threshold_ENTRY.delete(0, tk.END)
            self.lowpass_threshold_ENTRY.insert(0,self.lowpass_threshold)
            print(" Input requires integer values >= 0")
        ## check if input is in range
        if user_input >= 0:
            if DEBUG: print(" lowpass threshold updated: %s" % user_input )
            self.lowpass_threshold = user_input
            ## pass focus back to the main instance
            self.instance.focus()
            self.next_img('none')
        else:
            self.lowpass_threshold_ENTRY.delete(0, tk.END)
            self.lowpass_threshold_ENTRY.insert(0,self.lowpass_threshold)
            print(" Input requires positive integer values")
        return

    def sigma_updated(self):
        user_input = self.sigma_contrast_ENTRY.get().strip()
        ## cast the input to a float value
        try:
            user_input = float(user_input)
        except:
            self.sigma_contrast_ENTRY.delete(0, tk.END)
            self.sigma_contrast_ENTRY.insert(0,self.sigma_contrast)
            print(" Input requires float values >= 0")
        ## check if input is in range
        if user_input > 0:
            if DEBUG: print("sigma contrast updated to %s" % user_input )
            self.sigma_contrast = user_input
            ## pass focus back to the main instance
            self.instance.focus()
            self.next_img('none')
        else:
            self.sigma_contrast_ENTRY.delete(0, tk.END)
            self.sigma_contrast_ENTRY.insert(0,self.sigma_contrast)
            print(" Input requires positive float values")

        return

    def scalebar_updated(self):
        user_input = self.scalebar_length_ENTRY.get().strip()
        ## cast the input to an integer value
        try:
            user_input = int(user_input)
        except:
            self.scalebar_length_ENTRY.delete(0, tk.END)
            self.scalebar_length_ENTRY.insert(0,self.scalebar_length)
            print(" Input requires integer values > 0")
        ## check if input is in range
        if user_input > 0:
            if DEBUG: print("scalebar length updated %s" % user_input )
            self.scalebar_length = user_input
            ## pass focus back to the main instance
            self.instance.focus()
            self.next_img('none')
        else:
            self.scalebar_length_ENTRY.delete(0, tk.END)
            self.scalebar_length_ENTRY.insert(0,self.scalebar_length)
            print(" Input requires positive integer values")
        return

    def load_img(self, fname):
        """ Load the mrc image with the given file name
        """
        if DEBUG: speedtest('start')

        mrc_im_array, self.pixel_size = get_mrc_raw_data(fname)
        self.mrc_dimensions = (mrc_im_array.shape[1], mrc_im_array.shape[0])

        ###################################
        FAST = self.SPEED_OVER_ACCURACY.get()
        if not FAST:
            ## slow but accurate method
            img_array, ctf_img_array = mrc2grayscale(mrc_im_array, self.pixel_size, self.lowpass_threshold)
            img_scaled = resize_image(img_array, self.scale_factor)
            img_contrasted = sigma_contrast(img_scaled, self.sigma_contrast)
            im_obj = get_PhotoImage_obj(img_contrasted, self.SHOW_SCALEBAR.get(), scalebar_px = int(self.scalebar_length / (self.pixel_size / self.scale_factor)), scalebar_stroke = self.scalebar_stroke)
            ctf_scaled = resize_image(ctf_img_array, self.scale_factor)
            ctf_contrasted = sigma_contrast(ctf_scaled, self.sigma_contrast)
            ctf_contrasted = gamma_contrast(ctf_contrasted, 0.4)
            ctf_obj = get_PhotoImage_obj(ctf_contrasted)
        else:
            ## it is much faster if we scale down the image prior to doing filtering
            img_scaled = resize_image(mrc_im_array, self.scale_factor)
            img_array, ctf_img_array = mrc2grayscale(img_scaled, self.pixel_size / self.scale_factor, self.lowpass_threshold)
            img_contrasted = sigma_contrast(img_array, self.sigma_contrast)  
            im_obj = get_PhotoImage_obj(img_contrasted, self.SHOW_SCALEBAR.get(), scalebar_px = int(self.scalebar_length / (self.pixel_size / self.scale_factor)), scalebar_stroke = self.scalebar_stroke)
            ctf_contrasted = sigma_contrast(ctf_img_array, self.sigma_contrast)
            ctf_contrasted = gamma_contrast(ctf_contrasted, 0.4)
            ctf_obj = get_PhotoImage_obj(ctf_contrasted)
        ####################################

        ## update the display data on the class
        self.display_data = [ im_obj, ctf_obj ]
        self.display_im_arrays = [ img_contrasted, ctf_contrasted ]
        self.image_name = os.path.basename(fname)

        # a, b = get_fixed_array_index(1, 1)
        ## initialize a canvas if it doesnt exist yet
        if len(self.displayed_widgets) == 0:
            self.add_canvas(self.scrollable_frame, img_obj = im_obj, row = 0, col = 0, canvas_reference = self.displayed_widgets, img_reference = self.display_data)
        else:
            ## otherwise, just update the display data instead of creating a fresh canvas object
            if DEBUG: print(" redraw existing canvas")
            if self.SHOW_CTF.get() == True:
                self.load_img_on_canvas(self.displayed_widgets[0], self.display_data[1])
            else:
                self.load_img_on_canvas(self.displayed_widgets[0], self.display_data[0])

        ## update label/entry widgets
        self.update_input_widgets()

        ## draw image coordinates if necessary
        self.draw_image_coordinates()

        if DEBUG: speedtest('stop')
        return

    def debugging(self):
        # print(" %s coordinates in dictionary" % len(self.coordinates))
        # self.rescale_picked_coordinates(0.25, 0.2)
        # coord2freq(450, 450, self.display_data[0].width(), self.display_data[0].height(), self.pixel_size / self.scale_factor)
        return

    def select_all(self, widget):
        """ This function is useful for binding Ctrl+A with
            selecting all text in an Entry widget
        """
        return widget.select_range(0, tk.END)

    def update_input_widgets(self):
        """ Updates the input widgets on the main GUI to take on the values of the global dictionary.
            Mainly used after loading a new settings file.
        """
        self.image_name_label.delete(0, tk.END)
        self.image_name_label.insert(0,self.image_name)

        self.scale_ENTRY.delete(0, tk.END)
        self.scale_ENTRY.insert(0,self.scale_factor)

        self.lowpass_threshold_ENTRY.delete(0, tk.END)
        self.lowpass_threshold_ENTRY.insert(0,self.lowpass_threshold)

        self.sigma_contrast_ENTRY.delete(0, tk.END)
        self.sigma_contrast_ENTRY.insert(0,self.sigma_contrast)

        self.picks_diameter_ENTRY.delete(0, tk.END)
        self.picks_diameter_ENTRY.insert(0,self.picks_diameter)

        self.scalebar_length_ENTRY.delete(0, tk.END)
        self.scalebar_length_ENTRY.insert(0,self.scalebar_length)

        self.MRC_dimensions_LABEL['text'] = "(%s, %s)" % (self.mrc_dimensions)
        self.MRC_angpix_LABEL['text'] = "%s Å/px" % (self.pixel_size)
        self.MRC_displayed_angpix_LABEL['text'] = "Display @ %0.2f Å/px" % (self.pixel_size / self.scale_factor)

        # self.draw_image_coordinates()

        return

    def initialize_scrollable_window(self, viewport_frame):
        self.viewport_canvas = viewport_canvas = tk.Canvas(viewport_frame)
        self.viewport_scrollbar_y = viewport_scrollbar_y = ttk.Scrollbar(viewport_frame, orient="vertical", command=viewport_canvas.yview)
        self.viewport_scrollbar_x = viewport_scrollbar_x = ttk.Scrollbar(viewport_frame, orient="horizontal", command=viewport_canvas.xview)
        self.scrollable_frame = scrollable_frame = ttk.Frame(viewport_canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: viewport_canvas.configure(scrollregion=viewport_canvas.bbox("all")))

        viewport_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.viewport_canvas.configure(yscrollcommand=viewport_scrollbar_y.set, xscrollcommand=viewport_scrollbar_x.set)

        # viewport_frame.grid(row = 1, column = 0)
        # viewport_frame.pack()
        viewport_scrollbar_y.pack(side="right", fill="y")
        viewport_scrollbar_x.pack(side="bottom", fill="x")
        viewport_canvas.pack()#fill="both", expand= True)#(side="left", fill="both", expand=True)


        return scrollable_frame, viewport_canvas

    def initialize_menubar(self, parent):
        """ Create the top menu bar dropdown menus
        """
        ## initialize the top menu bar
        menubar = tk.Menu(parent)
        parent.config(menu=menubar)
        ## dropdown menu --> File
        dropdown_file = tk.Menu(menubar)
        menubar.add_cascade(label="File", menu = dropdown_file)
        dropdown_file.add_command(label="Open .mrc", command=self.load_file)
        dropdown_file.add_command(label="Save .jpg", command=self.save_jpg)
        dropdown_file.add_command(label="Exit", command=self.quit)
        # ## dropdown menu --> Options
        # dropdown_options = tk.Menu(menubar)
        # menubar.add_cascade(label="Options", menu = dropdown_options)
        return

    def resize(self, event):
        """ Sort through event callbacks and detect if the main UI window has been adjusted manually.
            If so, resize the viewport canvas to fit comfortably within the new dimensions.
        """
        ## find the identity of the widget passing in the event data
        widget = event.widget

        ## if the wiget is the main instance, then use its pixel dimensions to adjust the main canvas viewport window
        if widget == self.instance:

            # if DEBUG:
            #     print(" MainUI dimensions adjusted (%s, %s), resize viewport UI" % (event.height, event.width))

            ## determine offsets to comfortably fit the scrollbar on the right side of the main window
            # h = event.height - 5
            h = event.height - 50#- 18
            w = event.width - 30
            self.viewport_canvas.config(width=w - 140, height=h) ## add width to pad in the right-side panel
        return

    def determine_program_dimensions(self, data_frame):
        """ Use the screen size and data frame sizes to determine the best dimensions for the main UI.
        PARAMETERS
            data_frame = ttk.Frame object that holds the main display objects in the primary UI window
        """
        ## get the pixel size of the main data window under control of the scrollbars
        data_x, data_y = data_frame.winfo_width(), data_frame.winfo_height()
        ## get the resolution of the monitor
        screen_x = self.instance.winfo_screenwidth()
        screen_y = self.instance.winfo_screenheight()

        if DEBUG:
            print(" Data window dimensions = (%s, %s); Screen resolution = (%s, %s)" % (data_x, data_y, screen_x, screen_y))

        if data_x > screen_x:
            w = screen_x - 150
        else:
            w = data_x + 150

        if data_y > screen_y:
            h = screen_y - 250
        else:
            h = data_y + 30

        return w, h

    def canvas_callback(self, event):
        """ Tie events for specific canvases by adding this callback
        """
        if DEBUG: print (" Clicked ", event.widget, "at", event.x, event.y)

        for canvas_obj in self.displayed_widgets:
            if event.widget == canvas_obj:
                # print("MATCH = ", canvas_obj)
                # self.toggle_canvas(canvas_obj)
                self.on_left_mouse_down(event.x, event.y)
        return

    def add_canvas(self, parent, img_obj = None, row = 0, col = 0, canvas_reference = [], img_reference = []):
        """ Dynamically add canvases to the main display UI
        """
        ## prepare the tk.Canvas object
        c = tk.Canvas(parent, width = 150, height = 150, background="gray")

        if img_obj != None:
            ## add the object to the scope of the class by making it a reference to a variable
            img_reference.append(img_obj)
            self.load_img_on_canvas(c, img_obj)
        ## add a on-click callback
        c.bind("<ButtonPress-1>", self.canvas_callback)
        ## pack the widget into the data parent frame
        c.grid(row = row, column = col)

        ## add the canvas to the reference variable for the main UI
        canvas_reference.append(c)
        return

    def destroy_active_canvases(self):
        ## clear the canvas objects in memory
        for canvas_obj in self.displayed_widgets:
            # print(canvas_obj, img_obj)
            # canvas_obj.grid_remove()
            canvas_obj.destroy()

        ## clear the placeholder variable for these objects on the root object
        self.displayed_widgets = []
        return

    def resize_program_to_fit_screen_or_data(self):
        """ Update the widget displaying the data and check the best default screen size for the main UI, then adjust to that size
        """
        self.scrollable_frame.update() ## update the frame holding the data
        w, h = self.determine_program_dimensions(self.scrollable_frame) ## use the frame data & screen resolution to find a reasonable program size
        self.instance.geometry("%sx%s" % (w + 24, h + 24)) ## set the main program size using the updated values
        return

    def quit(self):
        if DEBUG:
            print(" CLOSING PROGRAM")
        sys.exit()

    def load_img_on_canvas(self, canvas, img_obj):
        """ PARAMETERS
                self = instance of class
                canvas = tk.Canvas object belonging to self
                img_obj = ImageTk.PhotoImage object belonging to self
        """
        ## place the image object onto the canvas
        # canvas.create_image(0, 0, anchor=tk.NW, image = img_obj)
        x,y = img_obj.width(), img_obj.height()

        canvas.delete('all')
        canvas.create_image(int(x/2) + 1, int(y/2) + 1, image = img_obj)
        ## resize canvas to match new image
        canvas.config(width=x - 1, height=y - 1)
        return


##########################
### RUN BLOCK
##########################
if __name__ == '__main__':
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    from tkinter.filedialog import asksaveasfilename
    from tkinter.messagebox import showerror
    from tkinter import ttk
    import numpy as np
    import os, sys
    import time
    try:
        from PIL import Image as PIL_Image
        from PIL import ImageTk
    except:
        print("Problem importing PIL, try installing Pillow via:")
        print("   $ pip install --upgrade Pillow")

    import mrcfile
    try:
        import cv2 ## for resizing images with a scaling factor
    except:
        print("Could not import cv2, try installing OpenCV via:")
        print("   $ pip install opencv-python")
    import scipy.ndimage as ndimage

    usage()

    ## parse the commandline in case the user added specific file to open, it will open the last one if more than one is given 
    start_index =  0
    for arg in sys.argv:
        ## find the files in the working directory
        image_list = get_mrc_files_in_dir(".")
        # print("ARG = ", arg, arg[-4:])
        if arg[-4:] == ".mrc":
            start_index = find_file_index(arg, image_list)
            # print(" ... match file (name, index) ->  (%s, %s)" % (arg, start_index) )


    root = tk.Tk()
    app = MainUI(root, start_index)
    root.mainloop()
