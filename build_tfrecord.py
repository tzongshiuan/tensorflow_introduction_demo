import tensorflow as tf
import cv2
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import get_file as gf
import get_file_2 as gf


def main():
    # 資料集的位置
    # train_dataset_dir = 'images'
    train_dataset_dir = 'mnist/data'

    # 取回所有檔案路徑
    images, labels = gf.get_file(train_dataset_dir)

    # 開始寫入 TRRecord 檔案
    # gf.convert_to_TFRecord(images, labels, 'train_record/tfrecords')
    # gf.convert_to_TFRecord(images, labels, 'train_record/tfrecords_gray')
    gf.convert_to_TFRecord(images, labels, 'train_record/tfrecords_mnist')


if __name__ == '__main__':
    main()