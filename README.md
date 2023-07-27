# img2bin
Extracts binary data from photos of calculator LCDs.

## Requirements
* Python 3.x  
* OpenCV for Python

## Usage
1. Run `python img2bin.py <fileName> <width> <height>` in a console.
	* `fileName` should be a picture of LCD displaying the data you want to extract;
	* `width` is the pixel count of the width of the data area(if there's only a 96x31 area showing the data you want, type `96`). This number _must_ be a multiple of 8;
	* `height` is the pixel count of the height of the date area.

2. A window should pop up, asking you to select the corners from the picture you provided.
	* Read the text in the console to learn how to use this window to pick the corners;
	* I don't know how to do corner detection, that's why it's manual at this moment.

3. After comfirming the corners, another 2 windows should appear, showing parameter trackbars and transformation result.
	* Drag the trackbars to adjust the parameters;
	* Check the transformation result in another window;
	* If the result matches what you see in the photo, follow the tips in the console.

4. Now you should get a file named `out.bin` in the directory, containing `width*height/8` bytes of data extracted from the LCD picture.
	* If you chose to save the result, it will **OVERWRITE** the existing file! I hope you didn't store your bitcoin wallet in a file named `out.bin`.

## Notes
* This piece of code runs okay with Windows 10, Python 3.8.9 and OpenCV for Python 4.7.0. I didn't test it on any other system so I don't know if it works unmodified with other configurations.
* I don't know much Python/DSP/OpenCV things, most of the code in the file are copied from the Internet.
* No warranty of any kind is provided(despite it doesn't mess with the OS), use this at your own risk.
