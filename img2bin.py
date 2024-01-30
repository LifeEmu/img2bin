# math functions
from math import *
from tkinter.tix import WINDOW
# numpy
import numpy as np
# OpenCV
import cv2 as cv
# command line arguments
import sys

# set output binary file name here
OUTPUT_FILENAME = "out.bin"

# Global variable definitions
WINDOW_NAME = "Click to set the corners"
DISPLAY_WINDOW_NAME = "Transformation result"
PARAMETER_WINDOW_NAME = "Parameters"
MouseX, MouseY = 0, 0
IsCoordValid = False
SkipSave = False

Kernel = np.ones((3, 3), dtype= np.uint8)
#Kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))


# Mouse callback function
# See https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html
def recordPoint(event, x, y, flags, param):
	global MouseX, MouseY, IsCoordValid
	if event == cv.EVENT_LBUTTONDOWN:
		MouseX, MouseY = x, y
		IsCoordValid = True


# Trackbar callback function
def refreshImage(pos):
	global Img
	imgCopy = np.copy(Img)
	blockSize = adjustSize(cv.getTrackbarPos("blockSize", PARAMETER_WINDOW_NAME), 3)
	gConst = cv.getTrackbarPos("Constant", PARAMETER_WINDOW_NAME)
	kernelSize = adjustSize(cv.getTrackbarPos("kernelSize", PARAMETER_WINDOW_NAME), 1)
	kernel = np.ones((kernelSize, kernelSize), dtype= np.uint8)
#	kernel = cv.getStructuringElement(cv.MORPH_CROSS, (kernelSize, kernelSize))

	imgCopy = cv.adaptiveThreshold(imgCopy, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, gConst)
#	retVal, imgCopy = cv.threshold(Img, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
#	imgCopy = cv.medianBlur(imgCopy, 5)
	imgCopy = cv.morphologyEx(imgCopy, cv.MORPH_CLOSE, kernel)

#	imgCopy = cv.resize(imgCopy, (4 * PixelPerRow, 4 * PixelPerCol))

	imgCopy = cv.resize(imgCopy, (1 * PixelPerRow, 1 * PixelPerCol), interpolation= cv.INTER_CUBIC)
	retVal, imgCopy = cv.threshold(imgCopy, 192, 255, cv.THRESH_BINARY)
	imgCopy = cv.resize(imgCopy, (4 * PixelPerRow, 4 * PixelPerCol), interpolation= cv.INTER_NEAREST)

	cv.imshow(DISPLAY_WINDOW_NAME, imgCopy)


# Transparency callback function
def blendImages(pos):
	overlayImg = cv.resize(ConvertedImg, clipSize, interpolation= cv.INTER_NEAREST)
	alpha = pos / 100
	result = cv.addWeighted(overlayImg, alpha, clippedImg, 1.0 - alpha, 0.0)
	cv.imshow(DISPLAY_WINDOW_NAME, result)


# return a valid size for convolution operations
def adjustSize(size, _min):
	if size < _min:
		return _min | 1
	return size | 1

# Calculate Euclidian distance of 2 points
def distanceE(p1, p2):
	return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Calculate length and height of a rectangular area
# takes an array of 4 2-element tuples, return an numpy matrix
def calcCorners(points):
	width = int((distanceE(points[0], points[1]) + distanceE(points[2], points[3])) / 2)
	height = int((distanceE(points[0], points[2]) + distanceE(points[1], points[3])) / 2)
#	return (width - width % PixelPerRow - 1, height - height % PixelPerCol - 1)
	return (width - 1, height - 1)

def destMatrix(dim):
	width, height = dim
	return np.float32([(0, 0), (width, 0), (0, height), (width, height)])


# ================
# Code starts here

if len(sys.argv) < 4:
	print("Usage: {} <imgName> <pixel per row> <pixle>\n".format(sys.argv[0]))
	exit()

# Unpack the parameters
ImgName, PixelPerRow, PixelPerCol = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


# Read the image
Img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
if Img is None:
	print("Unable to open image \"{}\".".format(ImgName))
	exit()


# Prepare the copy to be displayed
Corners = [(0, 0), (0, 0), (0, 0), (0, 0)]
cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
cv.setMouseCallback(WINDOW_NAME, recordPoint)


# Let the user choose the corners
index = 0
key = -1
cv.imshow(WINDOW_NAME, Img)
print("""
Left click on picture to select a point.
Press 'A' to select next point
Press 'D' to select last point
Press 'N' to start over
Press 'Y' to confirm""")
while True:
	imgCopy = None
	imgCopy = np.copy(Img)
	while (IsCoordValid == False) and (key == -1):
		key = cv.waitKey(100)

	if (key == ord('a')) and (index < 4):
		index += 1
	if (key == ord('d')) and (index > 0):
		index -= 1

	if (index < 4) and (IsCoordValid == True):
		Corners[index] = (MouseX, MouseY)

	if (key == ord('y')) and (index >= 3):
		break
	if key == ord('n'):
		index = 0

	for cur in range(0, 4):
		cv.line(imgCopy, Corners[cur], Corners[(cur + 1) % 4], (15, 63, 15), 2)
	cv.line(imgCopy, Corners[index % 4], Corners[index % 4], (191, 63, 15), 3)
	cv.imshow(WINDOW_NAME, imgCopy)

	print("Point number = {}\nCorners = {}".format(index, Corners))

	key = -1
	IsCoordValid = False


# Swap the points
Corners[2], Corners[3] = Corners[3], Corners[2]

clipSize = calcCorners(Corners)

# Adjust for perspective
transformMatrix = cv.getPerspectiveTransform(np.float32(Corners), destMatrix(clipSize))
Img = cv.warpPerspective(Img, transformMatrix, clipSize)

clippedImg = np.copy(Img)	# make a copy of clipped image for later use

print("Size = {}".format(clipSize))


# Create windows for parameters
cv.namedWindow(PARAMETER_WINDOW_NAME)
cv.createTrackbar("blockSize", PARAMETER_WINDOW_NAME, 7, 63, refreshImage)
cv.createTrackbar("Constant", PARAMETER_WINDOW_NAME, 7, 63, refreshImage)
cv.createTrackbar("kernelSize", PARAMETER_WINDOW_NAME, 1, 15, refreshImage)
print("Press 'Y' to convert, if the outcome looks good enough,\nPress 'N' if you don't want to save the result.")
while True:
	key = cv.waitKey(100)
	if key == ord('y'):
		break
	if key == ord('n'):
		SkipSave = True
		break

G_BlockSize = adjustSize(cv.getTrackbarPos("blockSize", PARAMETER_WINDOW_NAME), 3)
G_Constant = cv.getTrackbarPos("Constant", PARAMETER_WINDOW_NAME)
M_KernelSize = adjustSize(cv.getTrackbarPos("kernelSize", PARAMETER_WINDOW_NAME), 1)
cv.destroyWindow(PARAMETER_WINDOW_NAME)

# 9, 11 works well with screens with large portions of white areas
# 7, 7 works well with screens with little white blocks
ConvertedImg = cv.adaptiveThreshold(Img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, G_BlockSize, G_Constant)
#retVal, Img = cv.threshold(Img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)


# Eliminate the black dots
Kernel = np.ones((M_KernelSize, M_KernelSize), dtype= np.uint8)
#Kernel = cv.getStructuringElement(cv.MORPH_CROSS, (M_KernelSize, M_KernelSize))
ConvertedImg = cv.morphologyEx(ConvertedImg, cv.MORPH_CLOSE, Kernel)


# Resize the picture
ConvertedImg = cv.resize(ConvertedImg, (1 * PixelPerRow, 1 * PixelPerCol), interpolation= cv.INTER_CUBIC)
retVal, ConvertedImg = cv.threshold(ConvertedImg, 192, 255, cv.THRESH_BINARY)


# Display the final outcome
imgCopy = cv.resize(ConvertedImg, (4 * PixelPerRow, 4 * PixelPerCol), interpolation= cv.INTER_NEAREST)
cv.imshow(DISPLAY_WINDOW_NAME, imgCopy)


# Manual correction
cv.destroyWindow(WINDOW_NAME)
cv.namedWindow(DISPLAY_WINDOW_NAME, cv.WINDOW_AUTOSIZE)
cv.imshow(DISPLAY_WINDOW_NAME, imgCopy)
cv.setMouseCallback(DISPLAY_WINDOW_NAME, recordPoint)
cv.namedWindow(PARAMETER_WINDOW_NAME)
cv.createTrackbar("Transparency", PARAMETER_WINDOW_NAME, 50, 100, blendImages)

print("""
Manual pixel correction:
Click on pixels to flip it,
'Y' to continue.
""")
key = -1
while key != ord('y'):
	key = -1
	while (key == -1) and (IsCoordValid == False):
		key = cv.waitKey(100)

	if IsCoordValid == True:
		ConvertedImg[int(MouseY / clipSize[1] * PixelPerCol), int(MouseX / clipSize[0] * PixelPerRow)] ^= 0xFF
		blendImages(cv.getTrackbarPos("Transparency", PARAMETER_WINDOW_NAME))

	IsCoordValid = False


# Convert to binary data
BinData = []
for row in range(0, PixelPerCol):
	for offset in range(0, PixelPerRow, 8):	# assume 1BPP
		byte = 0
		for bit in range(0, 8):
			byte = (byte << 1) | ((ConvertedImg.item(row, offset + bit) & 1) ^ 1)
		print(" {:02x}".format(byte), end='')
		BinData.append(byte)
	print("")


# Write to file
if SkipSave == False:
	outFile = open(OUTPUT_FILENAME, "wb")
	outFile.write(bytes(BinData))
	outFile.close()


# ----Test purpose only----
#'''
#print("Press [ESC] to quit")
#while True:
#	key = cv.waitKey(100)
#	if key == 27:
#		break
#'''


cv.destroyAllWindows()
