import tensorflow as tf
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import struct
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filename = "./mnist/train-images.idx3-ubyte"
# filename = "./mnist/train-labels.idx1-ubyte"

binfile = open(filename, 'rb')
buf = binfile.read()

index = 0
# read 4 unsigned int with big-endian format
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')  # move the cursor

for image in range(0, numImages):
    # the image is 28*28=784 unsigned chars
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')  # move the cursor

    # create a np array to save the image
    im = np.array(im, dtype='uint8')
    im = im.reshape(28, 28)

    # # display the image
    # plt.imshow(im, cmap='gray')
    # plt.show()

    im = Image.fromarray(im)
    im.save("mnist/data/train_%s.bmp" % image, "bmp")