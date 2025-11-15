#!/usr/bin/env python3
# version beta

# 2021-03-10: First draft created by Alex Keszei

## TO DO: - Hide the draw up while holding down right click mouse button


"""
"""

##########################
### FUNCTION DEFINITIONS
##########################

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import os, string, sys, pathlib
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from PIL import ImageTk
# import Image function from PIL under the alias 'PIL' (to avoid clashing with the Tk built-in 'Image' function)
import PIL.Image as PIL

class Gui:
    def __init__(self, master):
        """ The initialization scheme provides the grid layout, global keybindings,
            and widgets that constitute the main GUI window
        """
        self.master = master
        master.title("Tk-based FFT analyser")

        ## Menu bar layout
        menubar = Menu(self.master)
        self.master.config(menu=menubar)
        ## add items to the menu bar
        dropdown_file = Menu(menubar)
        menubar.add_cascade(label="File", menu=dropdown_file)
        dropdown_file.add_command(label="Open image", command=self.load_image)
        dropdown_file.add_command(label="Exit", command=self.quit)

        ## Widgets
        self.canvas = Canvas(master, width = 300, height = 300, background="gray", cursor="cross red red")

        self.angpix_label = Label(master, font=("Helvetica", 12), text="Ang/pix")
        self.input_angpix = Entry(master, width=18, font=("Helvetica", 12))
        self.input_angpix.insert(END, "%s" % angpix)

        self.rotation_label = Label(master, font=("Helvetica", 12), text="Rotate img CCW (deg)")
        self.input_rotation = Entry(master, width=18, font=("Helvetica", 12))
        self.input_rotation.insert(END, "%s" % rotation)

        self.scale_label = Label(master, font=("Helvetica", 12), text="FFT img scale factor")
        self.input_scale = Entry(master, width=18, font=("Helvetica", 12))
        self.input_scale.insert(END, "%s" % scaling_factor)

        # self.img_size_label = Label(master, font=("Helvetica", 11), text="img pix (x,y):")
        self.img_size = Label(master, font=("Helvetica italic", 10), text="%s, %s" % image_dimensions)
        self.FFT_canvas = Canvas(master, width = 600, height = 600, background="gray")
        self.display_FFT = self.FFT_canvas.create_image(0, 0, anchor=NW, image="")

        ## Widget layout
        self.canvas.grid(row=1, column=0, columnspan=2, sticky=N)

        # self.img_size_label.grid(row=2, column=0, padx=5, pady=0, sticky=NE)
        self.img_size.grid(row=2, column=0, columnspan=2, padx=5, pady=0)

        self.angpix_label.grid(row=4, column=0, padx=5, pady=0, sticky=NE)
        self.input_angpix.grid(row=4, column=1, padx=5, pady=0, sticky=NW)

        self.rotation_label.grid(row=5, column=0, padx=5, pady=0, sticky=NE)
        self.input_rotation.grid(row=5, column=1, padx=5, pady=0, sticky=NW)

        self.scale_label.grid(row=0, column=2, padx=5, pady=0, sticky=W)
        self.input_scale.grid(row=0, column=3, padx=5, pady=0, sticky=W)

        self.FFT_canvas.grid(row=1, column=2, rowspan=50, columnspan = 10)


        ## Input mapping
        self.FFT_canvas.bind("<ButtonPress-1>", self.on_left_mouse_down)

        self.input_angpix.bind('<KP_Enter>', lambda event: self.read_input_widgets()) # numpad 'Return' key
        self.input_angpix.bind('<Return>', lambda event: self.read_input_widgets())

        self.input_rotation.bind('<KP_Enter>', lambda event: self.read_input_widgets()) # numpad 'Return' key
        self.input_rotation.bind('<Return>', lambda event: self.read_input_widgets())

        self.input_scale.bind('<KP_Enter>', lambda event: self.read_input_widgets()) # numpad 'Return' key
        self.input_scale.bind('<Return>', lambda event: self.read_input_widgets())

        self.master.protocol("WM_DELETE_WINDOW", self.quit)
        return

    def on_left_mouse_down(self, event):
        global image_dimensions, scaling_factor
        mouse_position = event.x, event.y
        print("Mouse pressed at position: x, y =", mouse_position[0], mouse_position[1])

        ## draw a visual representation of where the program is running its calculation:

        ## delete any pre-existing visual data
        self.FFT_canvas.delete('selected_coordinate')

        self.FFT_canvas.create_line( (mouse_position[0], 0), (mouse_position[0], self.FFT_canvas.winfo_height()), fill='red', width=1, tags='selected_coordinate') # line goes through the series of points (x0, y0), (x1, y1), … (xn, yn)
        self.FFT_canvas.create_line( (0, mouse_position[1]), (self.FFT_canvas.winfo_width(), mouse_position[1]), fill='red', width=1, tags='selected_coordinate') # line goes through the series of points (x0, y0), (x1, y1), … (xn, yn)

        # ## each coordinate is the center of a box, thus we need to offset by half the gif_box_width pixel length to get the bottom left and top right of the rectangle
        # box_size = 20
        # x0 = mouse_position[0] - box_size / 2
        # y0 = mouse_position[1] - box_size / 2
        # x1 = mouse_position[0] + box_size / 2
        # y1 = mouse_position[1] + box_size / 2
        # self.FFT_canvas.create_rectangle(x0, y0, x1, y1, outline='red', width=1, tags='selected_coordinate')

        ## calculate the angular distance of the picked coordinate from the center of the image
        # print("Image dimensions = (%s, %s)" % image_dimensions)
        FFT_image_center_coordinate = (int(image_dimensions[0] * scaling_factor / 2), int(image_dimensions[1] * scaling_factor / 2))
        # print("FFT image center coordinate = (%s, %s)" % FFT_image_center_coordinate)
        difference_vector = tuple(b - a for a, b in zip(mouse_position, FFT_image_center_coordinate))
        difference_vector_magnitude = np.linalg.norm(difference_vector) / scaling_factor
        # print("Vector magnitude in pixels = ", difference_vector_magnitude)
        ## convert the magnitude value into frequency value in units angstroms
        frequency = ( 1 / ( difference_vector_magnitude / image_dimensions[0] ) ) * angpix
        print("Frequency of radial pixel position () = " + "{:.2f}".format(frequency) + " Ang")

        self.FFT_canvas.create_text(mouse_position[0] + 5, mouse_position[1] - 4, font=("Helvetica", 14), text = "{:.2f}".format(frequency) + " Ang", fill='red', anchor = SW,  tags='selected_coordinate')

        ## draw a guiding line that shows the vector being measured from center of image to the mouse position
        self.FFT_canvas.create_line( FFT_image_center_coordinate, mouse_position, fill='yellow', width=1, tags='selected_coordinate') # line goes through the series of points (x0, y0), (x1, y1), … (xn, yn)

        ## draw a guiding circle to indicate the resolution ring being measured
        self.FFT_canvas.create_oval(FFT_image_center_coordinate[0] - int(difference_vector_magnitude * scaling_factor), FFT_image_center_coordinate[1] - int(difference_vector_magnitude * scaling_factor), FFT_image_center_coordinate[0] + int(difference_vector_magnitude * scaling_factor), FFT_image_center_coordinate[1] + int(difference_vector_magnitude * scaling_factor), dash = (7,4,2,4 ), width = 1, outline = 'red', tags='selected_coordinate') # Creates a circle or an ellipse at the given coordinates. It takes two pairs of coordinates; the top left and bottom right corners of the bounding rectangle for the oval.
        return

    def quit(self):
        print("========================= CLOSING PROGRAM =========================")
        print(" ")
        ## get errors if I use this program to simply view a directory... only write out with Ctrl+S
        ## write out a settings file before closing
        # self.save_settings()
        ## write out the list of marked files before closing
        # self.write_marked()
        ## write out the command necessary to rename the _CURATED.star file into _manpick.star file to ease-of-use (a typical next step after curation)
        ## close the program
        sys.exit()

    def read_input_widgets(self):
        global angpix, rotation, scaling_factor

        ## Angstroms per pixel widget
        try:
            angpix = float(self.input_angpix.get().strip())
            print("Updated angpix variable = ", angpix)
        except:
            self.input_angpix.delete(0,END)
            self.input_angpix.insert(0, "angpix")
            angpix = 1

        ## Degrees rotation
        try:
            rotation = float(self.input_rotation.get().strip())
            print("Updated rotation variable = ", rotation)
        except:
            self.input_rotation.delete(0,END)
            self.input_rotation.insert(0, "rotation")
            rotation = 0

        ## FFT image scale
        try:
            scaling_factor = int(self.input_scale.get().strip())
            ## limit the scaling factor to 10 to avoid memory meltdowns
            if scaling_factor > 10:
                scaling_factor = 10
            elif scaling_factor <= 0:
                scaling_factor = 1
            print("Updated scaling_factor variable = ", scaling_factor)
        except:
            self.input_scale.delete(0,END)
            self.input_scale.insert(0, "FFT scaling (integer)")
            scaling_factor = 1


        ## call a redraw of the image with the new set variables
        self.draw_image(file_w_path)
        return

    def load_image(self):
        """ Permits the system browser to be launched to select an image
            form a directory. Loads the directory and file into their
            respective variables and returns them
        """
        global file_w_path
        # See: https://stackoverflow.com/questions/9239514/filedialog-tkinter-and-opening-files
        fname = askopenfilename(parent=self.master, initialdir=".", title='Select file', filetypes=(("All files", "*.*"),
                                           ("Joint photographic experts group", "*.jpeg;*.jpg"),
                                           ("PNG", "*.png"),
                                           ("Graphics interchange format", "*.gif") ))


        if fname:
            ## check it is a suitable image we can work with:
            # extract file information from selection
            file_w_path = str(fname)
            file_dir, file_name = os.path.split(str(fname))
            print("File selected: "+ file_w_path)
            if self.is_image(file_name):
                self.draw_image(file_w_path)
            else:
                print("Inappropriate file selected: ", file_name)

    def draw_image(self, img_path):
        """
        """
        global image_dimensions, rotation

        # load image onto canvas object using PhotoImage
        pil_new_img = PIL.open(pathlib.Path(img_path)).rotate(rotation).convert(mode='L') ## convert to grayscale data as we load it

        # pass the pillow Image() object to the FFT function
        self.draw_FFT(pil_new_img)
        # update canvas size parameters with new image dimensions
        x,y = pil_new_img.size
        ## set the global values for other functions to use
        image_dimensions = (x, y)

        self.current_img = ImageTk.PhotoImage(pil_new_img)
        self.display = self.canvas.create_image(0, 0, anchor=NW, image=self.current_img)
        self.canvas.display = self.display

        # resize canvas to match new image
        x,y = self.current_img.width(), self.current_img.height()
        self.canvas.config(width=x, height=y)

        ## update widget displaying pixel size of .GIF file
        self.img_size.config(text="%s, %s px" % (x, y))

        ## redraw the canvases and widgets to fit the incoming image
        root.update()

    def draw_FFT(self, PIL_img):
        global scaling_factor, rotation


        ## load the Pillow Image() object as a data array so we can work on it with matplotlib
        im = asarray(PIL_img) ## see: https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/

        ## shifted fourier to center of image
        f = np.fft.fft2(im)
        f_shift = np.fft.fftshift(f)
        f_complex = f_shift
        f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6

        # we take the logarithm of the absolute value of f_complex, because f_abs has tremendously wide range.
        f_bounded = 20 * np.log(f_abs)

        ## convert data to grayscale based on the new range
        f_img = 255 * f_bounded / np.max(f_bounded)

        f_img = f_img.astype(np.uint8)

        img_from_array = PIL.fromarray(f_img)
        x,y = img_from_array.size

        rescaled_x = x * scaling_factor
        rescaled_y = y * scaling_factor
        FFT_canvas_padding = 0 ## so the user can clearly see the edge of the FFT image add a padding to the canvas (NOTE: complicates the maths, so turned this off)

        img_from_array = img_from_array.resize((rescaled_x, rescaled_y), PIL.LANCZOS) # (height, width)
        self.current_FFT_img = ImageTk.PhotoImage(img_from_array)
        self.display_FFT = self.FFT_canvas.create_image( FFT_canvas_padding, FFT_canvas_padding, anchor=NW, image=self.current_FFT_img) # first two parameters represent the position of the image center on the target canvas
        self.FFT_canvas.config(width=rescaled_x + (FFT_canvas_padding * 2), height=rescaled_y + (FFT_canvas_padding * 2))

        self.FFT_canvas.display = self.display_FFT

        return

    def is_image(self, file):
        """ For a given file name, check if it has an appropriate suffix.
            Returns True if it is a file with proper suffix (e.g. .gif)
        """
        image_formats = [".gif", ".png", ".jpg", ".jpeg"]
        for suffix in image_formats:
            if suffix in file:
                return True
        return False



##########################
### RUN BLOCK
##########################
if __name__ == '__main__':

    image_dimensions = (0, 0) # real-space image (width, height) in pixels
    scaling_factor = 2 # how much to scale up the FFT image for easier analysis
    angpix = 1
    rotation = 0
    file_w_path = ""

    root = Tk()
    app = Gui(root)
    root.mainloop()
