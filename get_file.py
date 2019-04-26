import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_file(file_dir):
    # The images in each subfolder
    global fileNames
    images = []
    # The subfolders
    sub_folders = []

    # Using "os.walk" function to grab all the files in each folder
    for dirPath, dirNames, fileNames in os.walk(file_dir):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))

        for name in dirNames:
            sub_folders.append(os.path.join(dirPath, name))

    # To record the labels of the image dataset
    labels = []
    count = 0
    for a_folder in sub_folders:
        n_img = len(os.listdir(a_folder))
        labels = np.append(labels, n_img * [count])
        count += 1

    sub_folders = np.array([images, labels])
    sub_folders = sub_folders.transpose()

    image_list = list(sub_folders[:, 0])
    label_list = list(sub_folders[:, 1])
    # image_list = list(sub_folders[0])
    # label_list = [os.path.splitext(name)[0] for name in fileNames]
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


# 轉Int64資料為 tf.train.Feature 格式
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 轉Bytes資料為 tf.train.Feature 格式
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_TFRecord(images, labels, filename):
    global image_raw
    n_samples = len(labels)
    TFWriter = tf.python_io.TFRecordWriter(filename)

    print('\nTransform start...')
    for i in np.arange(0, n_samples):
        try:
            image = cv2.imread(images[i])
            # image = Image.open(images[i])

            # color image
            # if i == 0:
            #     img = Image.open(images[i])
            #     plt.imshow(img)
            #     plt.show()

            # gray scale image
            # if i == 1:
            #     img = Image.open(images[i]).convert('L')
            #     array = np.asarray(img)
            #     plt.imshow(array, cmap='gray', vmin=0, vmax=255)
            #     plt.show()

            if image is None:
                print('Error image:' + images[i])
            else:
                # image_raw = image.tobytes()
                image_raw = image.tostring()
                # print(len(image_raw))

            label = int(labels[i])

            # 將 tf.train.Feature 合併成 tf.train.Features
            ftrs = tf.train.Features(
                feature={'label': int64_feature(label),
                         'image_raw': bytes_feature(image_raw)}
            )

            # 將 tf.train.Features 轉成 tf.train.Example
            example = tf.train.Example(features=ftrs)

            # 將 tf.train.Example 寫成 tfRecord 格式
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')

    TFWriter.close()
    print('Transform done!')
