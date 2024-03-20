 def mean_filter(image_path, kernel_size):

image cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) = # Pad the image to handle borders using zero-padding # This ensures that the filter can be applied to border pixels as well padded_image = cv2.copyMakeBorder (image, kernel_size//2,

kernel_size//2,

kernel_size//2,

kernel_size//2,

cv2.BORDER_CONSTANT)

# Create an empty image to store the filtered result filtered_image = np. zeros_like(image)

# Iterate over each pixel in the image for y in range(image.shape[0]):

for x in range(image.shape[1]):

# Extract the local window centered at the current pixel window = padded_image [y:y+kernel_size, x:x+kernel_size]

# Calculate the mean value of the window mean_value = np.mean (window)

# Assign the mean value to the corresponding pixel in the filtered image

filtered_image[y, x] = mean_value

filtered_image= convolve (filtered_image, gaussian_kernel_array)
[11:15 pm, 23/10/2023] Smruti Kiit: import cv2

import numpy as np import pandas as pd

import os

from scipy.ndimage import convolve

def gaussian_kernel(size, size_y=None):

size= int(size)

if not size_y:

size_y = size

else:

size_y= int(size_y)

x, y = np.mgrid [-size: size+1, -size_y:size_y+1] g= np.exp(-(x*2/float(size) + y*2/float(size_y))) return gg.sum()

# Make the Gaussian kernel by calling the function

kernel_size1 = 7

gaussian_kernel_array = gaussian_kernel (kernel_size1)