import tensorflow as tf
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from io import StringIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_image(index):
    return {
        0: 'mnist/test/0/3.jpg',
        1: 'mnist/test/0/10.jpg',
        2: 'mnist/test/1/137.jpg',
        3: 'mnist/test/1/203.jpg',
        4: 'mnist/test/2/119.jpg',
        5: 'mnist/test/2/280.jpg',
        6: 'mnist/test/7/4423.jpg',
        7: 'mnist/test/7/8651.jpg',
        8: 'mnist/test/8/3406.jpg',
        9: 'mnist/test/8/8661.jpg',
    }[index]

with tf.Session() as sess:
    ##################################################
    # load model #
    # save_path = "./model/test_model.meta"
    save_path = "./model/test_model_mnist.meta"

    # 使用 import_meta_graph 載入計算圖
    saver = tf.train.import_meta_graph(save_path)

    # 使用 restore 重建計算圖
    # saver.restore(sess, "./model/test_model")
    saver.restore(sess, "./model/test_model_mnist")

    # 取出集合內的值
    x = tf.get_collection("input")[0]
    y = tf.get_collection("output")[0]

    ##################################################

    # 讀一張影像
    # for i in range(24):
        # img = cv2.imread('images/1/%d.jpg' % i)
    # img = cv2.imread('images/0/3.jpg')
    # img = cv2.imread('mnist/test/0/3.jpg', 0)

    for i in range(10):
        # 辨識影像，並印出結果
        img = cv2.imread(get_image(i), 0)
        result = sess.run(y, feed_dict={x: img.reshape((-1, 28 * 28))})
        # result = sess.run(y, feed_dict={x: img.reshape((-1, 256 * 256 * 3))})
        print(result)
