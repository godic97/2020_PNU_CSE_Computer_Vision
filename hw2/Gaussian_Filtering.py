from PIL import Image
import numpy as np
import math

# Part1
# 1
# input: 3 4 5
# Create 1D Filter
def boxfilter(n):
    # If 'n' isn't odd number, throwing error
    assert n % 2 == 1, "Dimension must be odd"

    # Fill the array with 0.04
    arr = np.ones((n, n)) * 0.04

    return arr


# 2
# input: 0.3 0.5 1
# Create 1D Gaussian Filter
def gauss1d(sigma):
    # Round to Nearest odd number
    mod_sigma = round(sigma * 6)

    # If "mod_sigma" is even number, add 1 to make odd number
    mod_sigma = mod_sigma + 1 - (mod_sigma % 2)

    # Generate Gaussian Filter
    arr = np.arange(mod_sigma) + 0.5 - (mod_sigma / 2)
    arr = np.exp((-arr * arr) / (2 * sigma * sigma))

    # Nomalize
    arr = arr / np.sum(arr)

    return arr

# 3
# input: 0.5 1
#Create 2D Gaussian FIlter by outering 1D Gaussian Filter
def gauss2d(sigma):
    g1 = gauss1d(sigma)
    arr = np.outer(g1, g1)

    # Nomalize
    arr = arr / np.sum(arr)
    return arr

# 4-a
# input:
# Convolve array to filter
def convolve2d(array, filter):
    # Generate result
    arr = np.zeros(array.shape)

    # Zero padding to  array
    pad_size = (int) ((filter.shape[0] - 1) / 2)
    pad_array = np.pad(array, pad_width=pad_size, mode="constant", constant_values=0)

    # Reverse the filter to convolve
    filp_filter = np.flip(filter)

    # cross-correlation
    for i in range(pad_size, array.shape[0] + pad_size):
        for j in range(pad_size, array.shape[1] + pad_size):

            # Make sub-array for cross-correlation
            tmp = np.array(pad_array[(i - pad_size):(i + pad_size + 1), (j - pad_size):(j + pad_size + 1)])

            #cross-correlation
            tmp = tmp * filp_filter
            arr[i-pad_size][j-pad_size] = np.sum(tmp)

    return arr.astype("int")

# 4-b
# Convolve array and Gaussian Filter using sigma
def gaussconvolve2d(array, sigma):
    return convolve2d(array, gauss2d(sigma))

# 4-c
# input:0b_dog.bmp
# Load a image
im = Image.open("../resource/0b_dog.bmp")

# Convert to grey
im = im.convert('L')

# Convert Image to Numpy array
arr_im = np.asarray(im)

# Filtering
filtering_im = gaussconvolve2d(arr_im, 3)

# Convert Numpy array to Image
img = Image.fromarray(filtering_im.astype("uint8"))

# Part2
# 1
# input: 1a_bicycle.bmp
# Make low frequency image
img_bicycle = Image.open("../resource/3a_fish.bmp")

# Convert an Image to numpy array
arr_bicycle = np.asarray(img_bicycle)

# Store base image size
base_size = arr_bicycle.shape

low_bicycle = np.zeros(base_size)

# Generate low frequency image for each channel
for i in range(0, 3):
    low_bicycle[: , :, i] = gaussconvolve2d(arr_bicycle[:, :, i], 3)

# Convert array to image
low_bicycle = low_bicycle.astype("int")
filtering_bicycle = Image.fromarray(low_bicycle.astype("uint8"), mode="RGB")

# 2
# input: 1b_motorcycle.bmp
# Make high frequency image
img_motorcycle = Image.open("../resource/3b_submarine.bmp")

# Convert an image to numpy array
arr_motorcycle = np.asarray(img_motorcycle)
high_motorcycle = np.zeros(base_size)

# Generate high frequency image for each channel
for i in range(0, 3):
    high_motorcycle[:, :, i] = arr_motorcycle[:, :, i] - gaussconvolve2d(arr_motorcycle[:, :, i], 3)

high_motorcycle_b = high_motorcycle.astype("int") + 128
filtering_motocycle = Image.fromarray(high_motorcycle_b.clip(min=0, max=255).astype('uint8'), mode="RGB")

# 3
#input: 1a_bicycle.bmp, 1b_motorcycle
# Make Hybrid image
# Sum low frequency image and hig frequenct image
arr_hybrid = low_bicycle + high_motorcycle

# Convert array to image
arr_hybrid = arr_hybrid.astype("int").clip(min=0, max=255)
img_hybrid = Image.fromarray(arr_hybrid.astype("uint8"), mode="RGB")

img_hybrid.show()