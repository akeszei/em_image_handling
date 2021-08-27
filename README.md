# em_image_conversion
Scripts to convert from one image type to another, focusing on common EM image types.  

## Usages

#### mrc2jpg.py
Convert .MRC 2D EM images to .JPG,.PNG,.TIF, or .GIF formats.  

`mrc2jpg.py   input.mrc  output.jpg `

Options include binning and addition of a scalebar of a specified size (in Angstroms), e.g.:

`mrc2.jpg.py  input.mrc output.jpg  --bin 4 --scalebar 200 --angpix 1.94`

Batch mode is enabled in a similar flavor to `e2proc2d.py` by using glob matching and defining the output format with an `@` symbol:  

`mrc2jpg.py   *.mrc  @.jpg `

Since this script can output more than .JPG format, consider linking the script with alternate names for user clarity (e.g. `ln -s mrc2jpg.py mrc2png.py`).

-----
#### ser2mrc.py
Convert ThermoFischer TIA-generated .SER format 2D EM images to .MRC format.  

`ser2mrc.py   input.ser   output.mrc `

Optionally, can also output a binned .JPG image in addition to the .MRC file:

`ser2mrc.py input.ser  output.mrc --jpeg <binning value>`

Batch mode is enabled as above, where the basename of the .SER file is inherited as the name of the output .MRC file: 

`ser2mrc.py   *.ser   @.mrc `  


---

## Dependencies
The goal of these scripts are to provide straight-forward, flexible, scripts that have simple cross-platform dependencies. Most modules can be installed via `pip`, others are provided within the repo itself. 

#### NumPy
Calculation & array handling in python. See: (https://numpy.org/)

`pip install numpy`

#### Pillow (PIL fork)  
Handles image formats for .JPG, .PNG, .TIF, .GIF. See: (https://pillow.readthedocs.io/en/stable/)  

`pip install --upgrade Pillow`

#### mrcfile   
I/O parser for .MRC image format. See: (https://pypi.org/project/mrcfile/)/

`pip install mrcfile`


#### serReader.py  
A custom parser to read/write ThermoFisher TIA-generated .SER format. This is a backup copy from the original author Peter Ercius @ openNCEM (https://bitbucket.org/ercius/openncem/src/master/, also at https://github.com/ercius/openNCEM). This dependency is satisfied so long as it is <u>saved in the same directory as the running scripts</u>. 
