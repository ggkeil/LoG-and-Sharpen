# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

# we assume both image and kernel are numpy arrays
# basically image convolution
def imgfilter(image, kernel, scale = 1):
    # grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
    # the spatial dimensions should be odd (i.e. 3x3, 5x5, 7x7)
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
            k = (roi * kernel).sum()

			# store the convolved value in the output (x,y)-
			# coordinate of the output image
            output[y - pad, x - pad] = k
            
    if scale == 1:
        output = rescale_intensity(output, in_range=(0,255))
        output = (output * 255).astype("uint8")
    # return the output image
    return output

# outputs the gaussian kernel with sigma and kernel size provided
def gaussiankernel(sigma, kernelsize):
    k = 1 # you can change this if desired
    sum = 0
    kernel = np.zeros((kernelsize,kernelsize)) # Pre-allocate matrix
    for i in range(-(kernelsize // 2), (kernelsize // 2) + 1):
        for j in range(-(kernelsize // 2), (kernelsize // 2) + 1):
            kernel[i + (kernelsize // 2), j + (kernelsize // 2)] = k * np.exp(-((i*i + j*j) / (2. * sigma * sigma)))
            sum += kernel[i + (kernelsize // 2), j + (kernelsize // 2)]
            
    return kernel / sum

# creates a 3 x 3 45 degree isotropic kernel
# other Laplacian kernels are commented out in this function
def laplaciankernel():
    #kernel = np.array([[-10, -5, -2, -1, -2, -5, -10], [-5, 0, 3, 4, 3, 0, -5], [-2, 3, 6, 7, 6, 3, -2], [-1, 4, 7, 8, 7, 4, -1], [-2, 3, 6, 7, 6, 3, -2], [-5, 0, 3, 4, 3, 0, -5], [-10, -5, -2, -1, -2, -5, -10]])
    #kernel = np.array([[-4, -1, 0, -1, -4], [-1, 2, 3, 2, -1], [0, 3, 4, 3, 0], [-1, 2, 3, 2, -1], [-4, -1, 0, -1, -4]])
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    #kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    #kernel = np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]])
    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return kernel

# used for muliplying the LoG by the constant c
def sharpenimg(c, image):
    output = c * image
    output = rescale_intensity(output, in_range=(0,255))
    output = (output * 255).astype("uint8")
    return output

# User input area, run this with command prompt
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="C:\\Users\\eagle\\OneDrive for Business\\ECE\\ECE 5470\\LoG and Sharpen\\woman.png")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"]) # use this to change the input image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Output dimensions of Input Image
height, width = image.shape[:2]
print("Input Image Width: " + str(width))
print("Input Image Height: " + str(height))

sigma = input("Input the sigma you would like to use for the Gaussian filter: ")
kernelsize = input("Input the kernel size you would like to use for the Gaussian filter: ")
gaussianOutput = imgfilter(gray, gaussiankernel(int(sigma), int(kernelsize))) # blur the image using the grayscale image and gaussian kernel with 
laplacianOutput = imgfilter(gaussianOutput, laplaciankernel(), 0) # find edges of blurred image using the Laplacian kernel. Do not scale after Laplacian is applied to Gaussian

c = input("Input the constant you would like to multiply the LoG image: ")
sharpenedimg =  sharpenimg(int(c), laplacianOutput) # multiply the LoG image by a constant provided
imgSum = gray + sharpenedimg # add the original image to the LoG image multiplied by a constant

# show the output images
cv2.imshow("original", gray)
cv2.imshow("Output After Gaussian", gaussianOutput)
cv2.imshow("Output After Laplacian", laplacianOutput)
cv2.imshow("Output After Sharpening", imgSum)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("WomanAfterGaussian", gaussianOutput)
cv2.imwrite("WomanAfterLaplacian", laplacianOutput)
cv2.imwrite("WomanAfterSharpening", imgSum)