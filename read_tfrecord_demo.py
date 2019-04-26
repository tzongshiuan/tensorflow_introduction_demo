import tensorflow as tf
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from io import StringIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全部要讀取的文件名
filename = ["train_record/tfrecords"]

# 產生文件名隊列
filename_queue = tf.train.string_input_producer(filename,
                                                num_epochs=1,
                                                shuffle=True)

# 數據讀取器
reader = tf.TFRecordReader()
key, serialized_example = reader.read(filename_queue)

img_features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'image_raw': tf.FixedLenFeature([], tf.string)
                                   })
image = tf.decode_raw(img_features['image_raw'], tf.uint8)
image = tf.reshape(image, [256, 256, 3])
label = tf.cast(img_features['label'], tf.int64)

with tf.Session() as sess:
    # 初始化是必要的動作
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # 建立執行緒協調器
    coord = tf.train.Coordinator()

    # 啟動文件隊列，開始讀取文件
    threads = tf.train.start_queue_runners(coord=coord)

    count = 0
    try:
        # 讀 10 張影像
        while not coord.should_stop():
        # while count < 1:
            # 這邊讀取
            image_data, label_data = sess.run([image, label])

            # 這邊輸出
            # 因為已經經過解碼，二進制的資料已經轉換成影像檔，因此可以直接使用
            # 影像檔的方式輸出資料。
            cv2.imwrite('read_images/tf_%d_%d.jpg' % (label_data, count), image_data)
            count += 1

        print('Done!')

    except tf.errors.OutOfRangeError:
        print('Done!')

    finally:
        # 最後要記得把文件隊列關掉
        coord.request_stop()

    coord.join(threads)