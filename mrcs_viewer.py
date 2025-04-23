#!/usr/bin/env python3

## Written by: Alexander Keszei
## 2022-05-01: mrcs_viewer.py version 1 complete
## 2023-04-27: Updated to correctly display .mrcs files of raw exposures (i.e. tilt series). Also added other functionality.
## TO DO
## - Clean up code for scrollbar (pack it only in the initialize_canvases function, not also in __init__, no?)
## - Update the header info correctly! 

DEBUG = True

def usage():
    print("================================================================================================")
    print(" Open an .MRCS file and display frames as normalized grayscale images. Frames can be selected ")
    print(" by left-clicking and selected frames saved out into a new `subset.mrcs' file.")
    print("    $ mrcs_viewer.py  <input>.mrcs")
    print("------------------------------------------------------------------------------------------------")
    print(" Can set options on loadup rather than GUI via: ")
    print("            --scale (1) : rescale displayed frame images by a value in range (0,inf)")
    print("      --max_frames (50) : display only the first n frames from the input file")
    print("             --cols (8) : arrange displayed images in an array defined by this column shape")
    print("------------------------------------------------------------------------------------------------")
    print(" Keyboard shortcuts: ")
    print("    Ctrl + S : Save out a 'subsets.mrcs' file with the active selection")
    print("           i : Invert the current selection")
    print("    Ctrl + q : Quit program")
    print("================================================================================================")
    sys.exit()
    return

def make_empty_mrcs(stack_size, mrc_dimensions, dtype, fname):
    """ Prepare an empty .MRCS in memory of the correct dimensionality
    """
    with mrcfile.new(fname, overwrite=True) as mrcs:
        mrcs.set_data(np.zeros(( stack_size, ## stack size
                                mrc_dimensions[1], ## pixel height, 'Y'
                                mrc_dimensions[0]  ## pixel length, 'X'
                                ), dtype=np.dtype(getattr(np, str(dtype)))
                            ))

        ## set the mrcfile with the correct header values to indicate it is an image stack
        mrcs.set_image_stack()
        if DEBUG:
            print("======================================")
            print(" make_empty_mrcs ")
            print("--------------------------------------")
            print("  name = %s" % fname)
            print("  dimensions = (%s, %s, %s)" % (mrc_dimensions[0], mrc_dimensions[1], stack_size))
            print("  data type = %s" % dtype)
    return

def mrc2grayscale(mrc_raw_data):
    """ Convert raw mrc data into a grayscale numpy array suitable for display
    """
    ## remap the mrc data to grayscale range
    remapped = (255*(mrc_raw_data - np.min(mrc_raw_data))/np.ptp(mrc_raw_data)).astype(np.uint8) ## remap data from 0 -- 255 as integers
    return remapped

def sigma_contrast(im_array, sigma):
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

    if DEBUG:
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

def get_mrcs_info(fname):
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

    if DEBUG:
        print("======================================")
        print(" get_mrcs_info (%s) " % fname)
        print("--------------------------------------")
        print("  (x, y, z) = (%s, %s, %s)" % (x_dim, y_dim, z_dim) )
        print("  pixel_size = %s" % pixel_size)
        print("  dtype = %s " % dtype)

    return x_dim, y_dim, z_dim, pixel_size, dtype

def write_chosen_frames_to_empty_mrcs(input_mrcs_fname, chosen_frames, output_mrcs_fname):
    """ input_mrcs_fname = str(); file name in working dir of the parent .MRCS to take a subset from
        chosen_frames = list[int(), ..., int()]; list of integers of frames from the input mrcs to make a subset of
        output_mrcs_fname = str(); file name in working dir of the empty .MRCS we expect to write out
    """
    ## open the input mrcs into buffer
    input_mrcs = mrcfile.open(input_mrcs_fname, mode='r')
    output_mrcs = mrcfile.open(output_mrcs_fname, mode='r+')

    # print(" frames to save = ", chosen_frames)
    for i in range(len(chosen_frames)):
        ## grab the frame data we want to keep
        frame_num = chosen_frames[i]
        # print(" ... grabbing frame %s" % (frame_num + 1))
        ## sanity check there is a frame expected
        if frame_num in range(0, input_mrcs.data.shape[0]):
            frame_data = input_mrcs.data[frame_num]
            # if DEBUG:
                # print("Data read from file = (min, max) -> (%s, %s), dtype = %s" % (np.min(frame_data), np.max(frame_data), frame_data.dtype))

            ## need to deal with single frame as a special case since array shape changes format
            if len(chosen_frames) == 1:
                output_mrcs.data[0:] = frame_data
            else:
                ## pass the frame data into the next available frame of the output mrcs
                output_mrcs.data[i] = frame_data
                # if DEBUG:
                #     print("Data written to file = (min, max) -> (%s, %s), dtype = %s" % (np.min(output_mrcs.data[i]), np.max(output_mrcs.data[i]), output_mrcs.data[i].dtype))
        else:
            print(" Input frame value requested (%s) not in expected range of .MRCS input file: (%s; [%s, %s])" % (frame_num, input_mrcs_fname, 1, input_mrcs.data.shape[0]))

    output_mrcs.voxel_size = input_mrcs.voxel_size
    output_mrcs.update_header_from_data()
    output_mrcs.update_header_stats()

    output_mrcs.close()
    input_mrcs.close()
    return

def resize_image(im_array, scaling_factor):
    ## calculate the new dimensions based on the scaling factor and input image
    scaled_width = int(im_array.shape[1] * scaling_factor)
    scaled_height = int(im_array.shape[0] * scaling_factor)

    if DEBUG:
        print("======================================")
        print(" resize_image (scaling factor = %s)" % scaling_factor)
        print("--------------------------------------")
        print("  %s -> (%s, %s) " % (im_array.shape, scaled_width, scaled_height))

    # resized_im = cv2.resize(im_array, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST) ## for int arrays use INTER_NEAREST
    resized_im = cv2.resize(im_array, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA) ## for noisy micrographs, default INTER_LINEAR does not work well, switch to INTER_AREA

    return resized_im

def get_grayscale_img(raw_im, scaling_factor):
    remapped = mrc2grayscale(raw_im)
    remapped = sigma_contrast(remapped, 4)
    scaled = resize_image(remapped, scaling_factor)

    return scaled 

def get_mrcs_images(mrcs_file_path, scaling_factor, max_frames):
    # if DEBUG:
    #     print(" Grabbing frames (%s max) from .MRCS = %s" % (max_frames, mrcs_file_path))
    ## Unpack the data from the mrcs and pass it forward to the canvas objects
    img_stack = []

    ## open the mrcs file as an nparray of dimension (n, box_size, box_size), where n is the number of images in the stack
    with mrcfile.open(mrcs_file_path) as mrcs:
        counter = 0
        
        ## deal with single frame .mrcs files as a special case
        if len(mrcs.data.shape) == 2:
            processed_img = get_grayscale_img(mrcs.data.astype(np.float32), scaling_factor)
            img_stack.append(processed_img)
        else:
            ## interate over the mrcs stack by index n
            for n in range(mrcs.data.shape[0]):
                counter +=1
                if counter > max_frames:
                    return img_stack
                processed_img = get_grayscale_img(mrcs.data[n].astype(np.float32), scaling_factor)
                img_stack.append(processed_img)

    if True:
        print("======================================")
        print(" Extracted images from %s " % mrcs_file_path)
        print("--------------------------------------")
        print("   >> %s images extracted" % len(img_stack))
        print("   >> dimensions (x, y) = (%s, %s) pixels " % (img_stack[0].shape[0], img_stack[0].shape[1]))
        print("--------------------------------------")

    return img_stack

def get_imgs_in_dir(path):
    files_in_dir = os.listdir(path)
    images = []
    for file in files_in_dir:
        extension = file[-4:]
        if extension.lower() in [".jpg", ".png", ".gif"]:
            images.append(path + file)
    return images

def get_canvas_index(current_object_number, maximum_number_of_columns):
    row = int(current_object_number / maximum_number_of_columns)
    col = current_object_number % maximum_number_of_columns
    # print(" get_canvas_index -> (%s, %s)" % (row, col))
    return row, col

def get_PhotoImage_obj(im_array):
    PIL_img = PIL_Image.fromarray(im_array.astype(np.uint8))  #.convert('L')
    img_obj = ImageTk.PhotoImage(PIL_img)
    return img_obj

def create_image_array(imgs, ncols = 5, padding = 2, order_to_print = None):
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
        print("panel position (col, row) = (%s, %s)" % (col, row))

        y_range = (row * image_box_size + (PADDING * row), (row * image_box_size) + image_box_size + (PADDING * row))
        x_range = (col * image_box_size + (PADDING * col), (col * image_box_size) + image_box_size + (PADDING * col))

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

def add_scalebar(im, box_size, angpix, scalebar_size, indent_px = 8, stroke = 4):
    if angpix == 0:
        print(" ERROR :: Pixel size not known (currently detected as 0). Will not add scalebar.")
        return im

    scalebar_px = int(scalebar_size / angpix)
    if scalebar_px > box_size:
        print(" ERROR : Requested scalebar size (%s Ang, %s px) exceeds the dimensions of the image (%s px)!" % (scalebar_size, scalebar_px, box_size))
        usage()

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

    if DEBUG:
        print(" Printing scalebar onto first panel:")
        print("   >> %s pixels (%s Angstroms)" % (scalebar_px, scalebar_size))
        print("   >> Indent scalebar %s pixels from bottom left edge of panel" % indent_px)
        print("-------------------------------------------------------------")

    return im


class MainUI:
    def __init__(self, master, input_mrcs, input_scale, input_cols, input_max_frames):
        self.master = master
        master.resizable(False, False)
        master.title("MRCS handler")

        ## CLASS VARIABLES
        self.scaling_factor = input_scale
        self.max_columns = input_cols
        self.canvas_data = [] ## list of canvas objects known to the program and the image contents for each [ (tk.Canvas obj, ImageTk.PhotoImage obj), ... ]
        self.toggled_canvases = [] ## list of bool indicating which canvases are active (True) or not (False)
        # self.image_path = '' ## path to images
        self.input_mrcs = input_mrcs
        self.max_canvases = input_max_frames ## prevent loading an absurd amount of canvases on start up incase mrcs contains a shocking number of frames
        self.x_dim, self.y_dim, self.z_dim, self.pixel_size, self.dtype = get_mrcs_info(input_mrcs)

        ## MENU BAR LAYOUT
        ## initialize the top menu bar
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        ## file dropdown menu
        dropdown_file = tk.Menu(menubar)
        menubar.add_cascade(label="File", menu = dropdown_file)
        dropdown_file.add_command(label="Save subset.mrcs", command=self.save_selected_mrcs)
        dropdown_file.add_command(label="Save selected as .png", command = lambda: self.open_panel("save_png"))
        dropdown_file.add_command(label="Exit", command=self.quit)
        ## options dropdown menu
        dropdown_options = tk.Menu(menubar)
        menubar.add_cascade(label="Options", menu = dropdown_options)
        dropdown_options.add_command(label="Scaling factor", command = lambda: self.open_panel("scaling_factor"))
        dropdown_options.add_command(label="Number of columns", command = lambda: self.open_panel("column_number"))
        dropdown_options.add_command(label="Total frames to display", command = lambda: self.open_panel("max_canvases"))
        dropdown_options.add_command(label="Invert selection", command = lambda: self.invert_selection())

        ## SCROLL FUNCTIONALITY USING FRAMES & CANVASES
        ## estimate how large to make the working window on load up
        w, h = self.determine_program_dimensions()
        ## frame_viewport will be the frame that holds the viewport canvas object
        self.frame_viewport = tk.Frame(self.master)
        self.frame_viewport.grid(row=0, column=0)
        self.frame_viewport.rowconfigure(0, weight = 1)
        self.frame_viewport.columnconfigure(0, weight = 1)
        ## canvas_viewport will be the canvas object we use for scrolling functionality
        self.canvas_viewport = tk.Canvas(self.frame_viewport, width = w, height = h)
        self.canvas_viewport.grid(row = 0, column = 0, sticky = "nsew")
        ## frame_main will hold all the data images
        self.frame_main = tk.Frame(self.master)
        ## make frame_main a window of the main canvas viewport
        self.canvas_viewport.create_window(0, 0, window = self.frame_main, anchor="nw")

        ## set up the scrollbar object
        self.scrollbar = tk.Scrollbar(self.frame_viewport, orient=tk.VERTICAL)
        self.scrollbar.config(command=self.canvas_viewport.yview)
        self.canvas_viewport.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.grid(row = 0, column = 1, sticky = "ns")
        self.canvas_viewport.bind("<Configure>", self.update_scrollregion)

        self.initialize_canvases()

        ## KEYBINDINGS
        # self.master.bind("<F1>", lambda event: self.update_canvases())
        # self.master.bind("<F2>", lambda event: self.redraw_canvases())
        self.master.bind('<Control-KeyRelease-s>', lambda event: self.save_selected_mrcs())
        self.master.bind('<Control-KeyRelease-q>', lambda event: self.quit())
        self.master.bind('<i>', lambda event: self.invert_selection())

        ## Panel Instances
        self.optionPanel_instance = None
        return
    
    def invert_selection(self):
        ## iterate over the selection list and flip each value 
        for i in range(len(self.toggled_canvases)):
            existing_state = self.toggled_canvases[i]
            flipped_state = not existing_state
            self.toggled_canvases[i] = flipped_state


        if DEBUG:
            number_selected = 0
            for state in self.toggled_canvases:
                if state:
                    number_selected += 1
            print("======================================")
            print(" Invert selection " )
            print("--------------------------------------")
            print("  %s selected -> %s selected" % (len(self.toggled_canvases) - number_selected, number_selected))

        self.redraw_canvas_toggles()

        return 

    def open_panel(self, panelType = 'None'):
        ## use a switch-case statement logic to open the correct panel
        if panelType == "None":
            return
        elif panelType == "column_number":
            ## create new instance if none exists
            if self.optionPanel_instance is None:
                self.optionPanel_instance = OptionPanel(self, panelType, self.max_columns)
            ## otherwise, do not create an instance
            else:
                print(" OptionPanel is already open: ", self.optionPanel_instance)
        elif panelType == "max_canvases":
            ## create new instance if none exists
            if self.optionPanel_instance is None:
                self.optionPanel_instance = OptionPanel(self, panelType, self.max_canvases)
            ## otherwise, do not create an instance
            else:
                print(" OptionPanel is already open: ", self.optionPanel_instance)
        elif panelType == "scaling_factor":
            ## create new instance if none exists
            if self.optionPanel_instance is None:
                self.optionPanel_instance = OptionPanel(self, panelType, self.scaling_factor)
            ## otherwise, do not create an instance
            else:
                print(" OptionPanel is already open: ", self.optionPanel_instance)
        elif panelType == "save_png":
            ## create new instance if none exists
            if self.optionPanel_instance is None:
                self.optionPanel_instance = OptionPanel(self, panelType, self.scaling_factor)
            ## otherwise, do not create an instance
            else:
                print(" An OptionPanel is already open, close it first: ", self.optionPanel_instance)
        else:
            return
        return

    def determine_program_dimensions(self):
        mrc_dimensions_x, mrc_dimensions_y, mrc_dimensions_z = self.x_dim, self.y_dim, self.z_dim 
        scaled_frame_x = int(mrc_dimensions_x * self.scaling_factor)
        window_x = scaled_frame_x * self.max_columns + 50 ## add padding right to fit scrollbar
        ## get the resolution of the monitor
        screen_x = self.master.winfo_screenwidth()
        screen_y = self.master.winfo_screenheight()

        if window_x > screen_x:
            w = screen_x - 50
        elif mrc_dimensions_z < self.max_columns:
            w = scaled_frame_x * mrc_dimensions_z + 50
        else:
            w = window_x

        if mrc_dimensions_z > self.max_canvases:
            ## just use max canvas number to find the max rows
            total_rows = get_canvas_index(self.max_canvases, self.max_columns)[0] + 1
        else:
            ## otherwise we need to find the expected row number from the file dimensions
            total_rows = get_canvas_index(mrc_dimensions_z, self.max_columns)[0] + 1
        scaled_frame_y = int(mrc_dimensions_y * self.scaling_factor)
        window_y = scaled_frame_y * total_rows + 10 ## keep a bit of breathing room at bottom

        if window_y > screen_y:
            h = screen_y - 200
        else:
            h = window_y

        return w, h

    def update_scrollregion(self, event):
        self.canvas_viewport.configure(scrollregion=self.canvas_viewport.bbox("all"))

    def save_selected_mrcs(self):
        output_mrcs_name = 'subset.mrcs'
        ## sanity check the output mrcs name does NOT match the input name, or else it will output an empty file!
        if output_mrcs_name == os.path.basename(self.input_mrcs):
            print(" ERROR : Cannot both open & write out `subset.mrcs', rename input file and try again.")
            return
        mrc_dimensions = (self.x_dim, self.y_dim, self.z_dim) 
        ## determine the output mrcs stack size
        stack_size = 0
        frame_num = 0
        chosen_frames = []
        for switch in self.toggled_canvases:
            if switch == True:
                stack_size += 1
                chosen_frames.append(frame_num)
            frame_num += 1

        ## determine the dtype of the array 
        input_dtype = self.dtype 

        if stack_size > 0:
            make_empty_mrcs(stack_size, (mrc_dimensions[0], mrc_dimensions[1]), input_dtype, output_mrcs_name)
            write_chosen_frames_to_empty_mrcs(self.input_mrcs, chosen_frames, output_mrcs_name)

            print("======================================")
            print(" Written %s frames to: %s" % (stack_size, output_mrcs_name))
            print("--------------------------------------")
        else:
            print("======================================")
            print(" No frames were selected! No subset.mrcs file will be created...")
            print("--------------------------------------")

        return

    def save_selected_mrcs_as_img(self, columns, SHOW_SCALEBAR, scalebar_size_ang, SAVE_AS_INDIVIDUAL_IMGS):
            
        if SHOW_SCALEBAR:
            output_img_name = 'subset_%sApix_%sAngBar.png' % (self.pixel_size, scalebar_size_ang)
        else:
            output_img_name = 'subset_%sApix.png' % self.pixel_size
        print(" Output .png filename = %s" % output_img_name)

        # mrc_dimensions = (self.x_dim, self.y_dim, self.z_dim) 

        ## determine the output mrcs stack size
        stack_size = 0
        frame_num = 0
        chosen_frames = []
        for switch in self.toggled_canvases:
            if switch == True:
                stack_size += 1
                chosen_frames.append(frame_num)
            frame_num += 1

        # ## determine the dtype of the array 
        # input_dtype = self.dtype 

        imgs_list = []
        if stack_size > 0:
            for i in range(len(chosen_frames)):
                ## grab the frame data we want to keep
                frame_num = chosen_frames[i]
                print(" ... grabbing frame %s" % (frame_num + 1))

                ## sanity check there is a frame expected
                if frame_num in range(0, len(self.canvas_data)):
                    frame_canvas, frame_photoimage_obj = self.canvas_data[frame_num]
                    img = ImageTk.getimage( frame_photoimage_obj ).convert('L')
                    # img.show()
                    # img.close()
                    imgs_list.append(np.asarray(img))

        ## if no stack size specified, then take all 2d classes forward 
        else:
            for frame_num in range(0, len(self.canvas_data)):
                frame_canvas, frame_photoimage_obj = self.canvas_data[frame_num]
                img = ImageTk.getimage( frame_photoimage_obj ).convert('L')
                imgs_list.append(np.asarray(img))

        ## once we have accumulated all the images we want to write out, determine if we should save them in sequence or together as an array before publishing 
        if SAVE_AS_INDIVIDUAL_IMGS:
            ## modify the name to allow me to add each frame count to it
            output_img_basename = os.path.splitext(output_img_name)[0]
            count = 0
            for img_array in imgs_list:
                output_name = output_img_basename + "_" + str(count) + ".png"
                count += 1
                if SHOW_SCALEBAR:
                    img_array = add_scalebar(img_array, img_array.shape[0], self.pixel_size, scalebar_size_ang)
                im = PIL_Image.fromarray(img_array).convert('RGBA')
                im.save(output_name)
                print(" Saved image: ")
                print("   >> %s" % (output_name))


        else:
            display_img_as_array = create_image_array(imgs_list, ncols = columns)
            if SHOW_SCALEBAR:
                display_img_as_array = add_scalebar(display_img_as_array, imgs_list[0].shape[0], self.pixel_size, scalebar_size_ang)


            im = PIL_Image.fromarray(display_img_as_array).convert('RGBA')
            im.save(output_img_name)
            # if DEBUG:
            print(" Saved image: ")
            print("   >> %s" % output_img_name)
            print("-------------------------------------------------------------")
            im.show()

        return

    def toggle_canvas(self, canvas_obj):
        ## get the index of the canvas
        index = 0
        for c, img in self.canvas_data:
            if canvas_obj == c:
                break
            else:
                index += 1

        ## use the index to flip the correct canvas toggled status
        current_toggled_state = self.toggled_canvases[index]
        new_toggled_state = not current_toggled_state
        # print(" canvas obj ", canvas_obj, " has been toggled:", current_toggled_state, " -> ", new_toggled_state)
        self.toggled_canvases[index] = new_toggled_state

        # print(self.toggled_canvases)
        self.redraw_canvas_toggles()

        ## return focus to the master 
        self.master.focus
        return

    def redraw_canvas_toggles(self):
        """ Redraw selected canvases using toggle metadata
        """
        n = 0
        for canvas_obj, image_obj in self.canvas_data:
            x = image_obj.width()
            y = image_obj.height()
            INSET = 12
            WIDTH = 4
            ## delete any markers that may have previously existed
            canvas_obj.delete('markup')
            ## only draw rects on canvases marked as True in the root toggled_canvases variable
            if self.toggled_canvases[n] == True:
                canvas_obj.create_rectangle(x - INSET + int(WIDTH /2) + 1, y - INSET + int(WIDTH/2) + 1, INSET, INSET, outline='red', width = WIDTH, tags='markup')
            # print(x, y, canvas_obj)
            n += 1
        return

    def initialize_canvases(self):
        """ Draw the canvases and initialize the main variables
        """
        self.destroy_active_canvases(self.canvas_data)
        self.update_canvases()

        ## initialize the toggled canvas metadata variable
        self.toggled_canvases = []
        for x in range(len(self.canvas_data)):
            self.toggled_canvases.append(False)

        ## update scrollbar and data frame sizes
        w, h = self.determine_program_dimensions()
        self.canvas_viewport.config(width = w, height = h)
        self.canvas_viewport.grid(row = 0, column = 0, sticky = "nsew")
        self.canvas_viewport.update_idletasks()
        self.canvas_viewport.config(scrollregion=self.frame_main.bbox())

        return

    def update_canvases(self):
        """
        """
        mrcs_file_path = self.input_mrcs
        imgs = get_mrcs_images(mrcs_file_path, self.scaling_factor, self.max_canvases)
        for im_array in imgs:
            self.add_canvas(im_array, self.canvas_data, self.max_columns, self.toggled_canvases)

        return

    def canvas_callback(self, event):
        # print ("clicked ", event.widget, " at", event.x, event.y)

        for canvas_obj, image_obj in self.canvas_data:
            if event.widget == canvas_obj:
                # print("MATCH = ", canvas_obj)
                self.toggle_canvas(canvas_obj)
        return

    def add_canvas(self, img_array, canvas_data, max_columns, toggled_data):
        ## prepare the tk.Canvas object
        # c = tk.Canvas(self.master, width = 150, height = 150, background="gray", cursor="cross red red")
        c = tk.Canvas(self.frame_main, width = 150, height = 150, background="gray")
        c.bind("<ButtonPress-1>", self.canvas_callback)
        ## prepare the ImageTk.PhotoImage object
        img_obj = get_PhotoImage_obj(img_array)
        ## keep track of the new canvas to using the correct root tk variable
        canvas_data.append((c, img_obj))

        self.load_img_on_canvas(c, img_obj)
        ## pack the new canvas based on how many are currently active
        row_position, column_position = get_canvas_index(len(canvas_data) - 1, max_columns)
        c.grid(row = row_position, column = column_position)
        return

    def destroy_active_canvases(self, canvas_data):
        ## clear the canvas objects in memory
        for canvas_obj, img_obj in canvas_data:
            # print(canvas_obj, img_obj)
            # canvas_obj.grid_remove()
            canvas_obj.destroy()

        ## clear the placeholder variable for these objects on the root object
        self.canvas_data = []

        return

    def quit(self):
        if DEBUG:
            print(" Closing program...")
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

        canvas.create_image(int(x/2) + 1, int(y/2) + 1, image = img_obj)
        ## resize canvas to match new image
        canvas.config(width=x - 1, height=y - 1)
        return

class OptionPanel(MainUI):
    """ Panel GUI for manipulating specific options where the user inputs a value into an entry widget
    """
    def __init__(self, mainUI, option_type, option_variable): # Pass in the main window as a parameter so we can access its methods & attributes
        self.mainUI = mainUI # Make the main window an attribute of this Class object
        self.panel = tk.Toplevel() # Make this object an accessible attribute
        self.panel.resizable(False, False)
        self.panel.title('Option menu')
        self.SHOW_SCALEBAR = tk.BooleanVar(self.panel, True)
        self.SAVE_AS_INDIVIDUAL_IMGS = tk.BooleanVar(self.panel, False)

        if DEBUG:
            print(" Open option panel of type = ", option_type)
        if option_type == "column_number":
            self.input_label = tk.Label(self.panel, font=("Helvetica", 12), text="Column number: ")
            self.input_text = tk.Entry(self.panel, width=5, font=("Helvetica", 12), highlightcolor="blue", borderwidth=None, relief=tk.FLAT, foreground="black", background="light gray")
            self.apply_button = tk.Button(self.panel, text="Apply", command= lambda: self.update_column_number(self.input_text.get()), width=8)

            ## Pack widgets
            self.input_label.grid(column=0, row=0)#, sticky = tk.W)
            self.input_text.grid(column=1, row=0)#, sticky = tk.W)
            self.apply_button.grid(column=2, row=0) #, sticky = tk.W)

            ## Add some hotkeys for ease of use
            self.panel.bind('<Return>', lambda event: self.update_column_number(self.input_text.get()))
            self.panel.bind('<KP_Enter>', lambda event: self.update_column_number(self.input_text.get())) # numpad 'Return' key

            ## Set focus to the entry widget
            self.input_text.focus()


        if option_type == "max_canvases":
            self.input_label = tk.Label(self.panel, font=("Helvetica", 12), text="# Frames to display: ")
            self.input_text = tk.Entry(self.panel, width=5, font=("Helvetica", 12), highlightcolor="blue", borderwidth=None, relief=tk.FLAT, foreground="black", background="light gray")
            self.apply_button = tk.Button(self.panel, text="Apply", command= lambda: self.update_max_frames(self.input_text.get()), width=8)

            ## Pack widgets
            self.input_label.grid(column=0, row=0)#, sticky = tk.W)
            self.input_text.grid(column=1, row=0)#, sticky = tk.W)
            self.apply_button.grid(column=2, row=0) #, sticky = tk.W)

            ## Add some hotkeys for ease of use
            self.panel.bind('<Return>', lambda event: self.update_max_frames(self.input_text.get()))
            self.panel.bind('<KP_Enter>', lambda event: self.update_max_frames(self.input_text.get())) # numpad 'Return' key

            ## Set focus to the entry widget
            self.input_text.focus()


        if option_type == "scaling_factor":
            self.input_label = tk.Label(self.panel, font=("Helvetica", 12), text="Scaling factor: ")
            self.input_text = tk.Entry(self.panel, width=5, font=("Helvetica", 12), highlightcolor="blue", borderwidth=None, relief=tk.FLAT, foreground="black", background="light gray")
            self.apply_button = tk.Button(self.panel, text="Apply", command= lambda: self.update_scaling_factor(self.input_text.get()), width=8)

            ## Pack widgets
            self.input_label.grid(column=0, row=0)#, sticky = tk.W)
            self.input_text.grid(column=1, row=0)#, sticky = tk.W)
            self.apply_button.grid(column=2, row=0) #, sticky = tk.W)

            ## Add some hotkeys for ease of use
            self.panel.bind('<Return>', lambda event: self.update_scaling_factor(self.input_text.get()))
            self.panel.bind('<KP_Enter>', lambda event: self.update_scaling_factor(self.input_text.get())) # numpad 'Return' key

            ## Set focus to the entry widget
            self.input_text.focus()


        if option_type == "save_png":
            ## Generate widgets 
            self.save_individual_imgs_TOGGLE = tk.Checkbutton(self.panel, text='Save as separate imgs?', variable=self.SAVE_AS_INDIVIDUAL_IMGS, onvalue=True, offvalue=False)
            self.label_columns = tk.Label(self.panel, font=("Helvetica", 12), text="Columns: ")
            self.entry_columns = tk.Entry(self.panel, width=5, font=("Helvetica", 12), highlightcolor="blue", borderwidth=None, relief=tk.FLAT, foreground="black", background="light gray")
            self.label_scalebar = tk.Label(self.panel, font=("Helvetica", 12), text="Scalebar (A): ")
            self.entry_scalebar = tk.Entry(self.panel, width=5, font=("Helvetica", 12), highlightcolor="blue", borderwidth=None, relief=tk.FLAT, foreground="black", background="light gray")
            self.show_picks_TOGGLE = tk.Checkbutton(self.panel, text='Show scalebar?', variable=self.SHOW_SCALEBAR, onvalue=True, offvalue=False) 
            self.button_save_img = tk.Button(self.panel, text="Save .png", command= lambda: self.save_selected_as_png(), width=8)

            ## set defaults 
            self.entry_columns.insert(-1, '6')
            self.entry_scalebar.insert(-1, '250')

            ## Pack widgets
            self.save_individual_imgs_TOGGLE.grid(column=0, row=0)
            self.label_columns.grid(column=0, row=1)#, sticky = tk.W)
            self.entry_columns.grid(column=1, row=1)#, sticky = tk.W)
            self.show_picks_TOGGLE.grid(column=0, row = 2)
            self.label_scalebar.grid(column = 0, row = 3)
            self.entry_scalebar.grid(column = 1, row = 3)
            self.button_save_img.grid(column=2, row=4) #, sticky = tk.W)

            ## Add some hotkeys for ease of use
            # self.panel.bind('<Return>', lambda event: self.update_scaling_factor(self.input_text.get()))
            # self.panel.bind('<KP_Enter>', lambda event: self.update_scaling_factor(self.input_text.get())) # numpad 'Return' key


        ## add an exit function to the closing of this top level window
        self.panel.protocol("WM_DELETE_WINDOW", self.close)
        return

    def update_column_number(self, input_val):
        if int(input_val) > 0:
            self.mainUI.max_columns = int(input_val)
            if DEBUG:
                print(" ... updated column number to: ", input_val)
            self.mainUI.initialize_canvases()
            self.close()
        return

    def update_max_frames(self, input_val):
        if int(input_val) > 0:
            self.mainUI.max_canvases = int(input_val)
            if DEBUG:
                print(" ... updated max frames to: ", input_val)
            self.mainUI.initialize_canvases()
            self.close()
        return

    def update_scaling_factor(self, input_val):
        if float(input_val) > 0:
            self.mainUI.scaling_factor = float(input_val)
            if DEBUG:
                print(" ... updated scaling factor to: ", input_val)
            self.mainUI.initialize_canvases()
            self.close()
        return

    def save_selected_as_png(self):
        ## get the number of columns 
        try:
            columns = int(self.entry_columns.get())
        except:
            print(" Could not parse input to Columns widget (%s), try using an integer" % self.entry_columns.get())
        ## get the boolean for scalebar 
        SHOW_SCALEBAR = self.SHOW_SCALEBAR.get()
        ## if true, get the size of the desired scalebar
        try:
            scalebar_size_ang = int(self.entry_scalebar.get())
        except:
            print(" Could not parse input to scalebar size widget (%s), try using an integer" % self.entry_scalebar.get())

        ## get the boolean if imgs should be saved as separate imgs
        SAVE_AS_INDIVIDUAL_IMGS = self.SAVE_AS_INDIVIDUAL_IMGS.get()

        print(" Columns = %s " % columns, type(columns))
        print(" Show scalebar = %s" % SHOW_SCALEBAR, type(SHOW_SCALEBAR))
        print(" Scalebar size (ang) = %s" % scalebar_size_ang, type(scalebar_size_ang))
        self.mainUI.save_selected_mrcs_as_img(columns, SHOW_SCALEBAR, scalebar_size_ang, SAVE_AS_INDIVIDUAL_IMGS)
        return 

    def close(self):
        ## unset the panel instance before destroying this instance
        self.mainUI.optionPanel_instance = None
        ## destroy this instance
        self.panel.destroy()
        return


##########################
### RUN BLOCK
##########################
if __name__ == '__main__':
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    from tkinter.messagebox import showerror
    from tkinter import ttk
    import numpy as np
    import os, string, sys, math
    from PIL import Image as PIL_Image
    from PIL import ImageTk
    import re ## for use of re.findall() function to extract numbers from strings
    import copy
    import mrcfile
    import cv2 ## for resizing images with a scaling factor
    import cmdline_parser

    ##################################
    ## ASSIGN DEFAULT VARIABLES
    ##################################
    PARAMS = {
        'input_mrcs' : str(),
        'scale' : 1,
        'max_frames' : 50,
        'columns' : 8
        }
    ##################################

    ##################################
    ## SET UP EXPECTED DATA FOR PARSER
    ##################################
    FLAGS = {
 ##      flag      :  (PARAMS_key,     data_type,  legal_entries/range,   toggle for entry,   intrinsic toggle,  has_defaults)
    '--scale'      :  ('scale'   ,       float(),         (0.001, 999),              False,              False,         True ),
    '--max_frames' :  ('max_frames',       int(),           (1, 99999),              False,              False,         True ),
    '--cols'       :  ('columns',          int(),             (1, 999),              False,              False,         True )
    }


    FILES = {            ## cmd line index    allowed extensions                          ## can launch batch mode
        'input_mrcs' : (     1,               '.mrcs',                                     False)
        }
    ##################################

    PARAMS, EXIT_CODE = cmdline_parser.parse(sys.argv, 1, PARAMS, FLAGS, FILES)
    if EXIT_CODE < 0:
        usage()
        sys.exit()
    if DEBUG:
        cmdline_parser.print_parameters(PARAMS, sys.argv)

    ## Get the execution path of this script so we can find local modules
    # script_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    root = tk.Tk()
    app = MainUI(root, PARAMS['input_mrcs'], PARAMS['scale'], PARAMS['columns'], PARAMS['max_frames'])
    root.mainloop()
