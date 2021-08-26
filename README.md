# em_image_conversion
Scripts to convert from one image type to another for common EM file types.

<b>mrc2jpg.py</b> = Convert .MRC 2D EM images to .JPG format. Alternative output formats can be used (.PNG, .TIF, .GIF). Options include binning and addition of a scalebar of specified size. Dependencies include mrcfile, and PIL. 
> mrc2jpg.py   input.mrc   output.jpg   <br />
> mrc2jpg.py   \*.mrc   @.jpg  # batch mode 
-----
<b>ser2mrc.py</b> = Script to convert ThermoFischer TIA-generated .SER format 2D EM images to .MRC format. Optionally, can output binned .jpeg images as it runs. Dependencies include mrcfile, PIL, and the serReader.py scipt, which must be in the same directory as this script to be found by default. 
> ser2mrc.py   input.ser   output.mrc  <br />
> ser2mrc.py   \*.ser   @.mrc  # batch mode 
-----

<b>serReader.py</b> = A dependency for all ThermoFisher TIA-generated .SER file handling. This is a backup copy from the original author Peter Ercius @ openNCEM (https://bitbucket.org/ercius/openncem/src/master/)
