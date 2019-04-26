import tensorflow as tf
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from io import StringIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_width = 28
img_height = 28
is_gray = True


def read_and_decode(filename, batch_size):
    # 產生文件名隊列
    filename_queue = tf.train.string_input_producer(filename,
                                                    num_epochs=None,
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
    if is_gray:
        image = tf.reshape(image, [img_width, img_height, 1])
    else:
        image = tf.reshape(image, [img_width, img_height, 3])

    label = tf.cast(img_features['label'], tf.int64)

    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=10000 + 3 * batch_size,
        min_after_dequeue=1000)

    return image_batch, label_batch

    # tf.train.shuffle_batch 重要參數說明
    # tensors：   排列的張量。
    # batch_size：從隊列中提取新的批量大小。
    # capacity：  一個整數。隊列中元素的最大數量。
    # min_after_dequeue：出隊後隊列中的最小數量元素，用於確保元素的混合級別。


# if is_gray:
#     filename = ["train_record/tfrecords_gray"]
# else:
#     filename = ["train_record/tfrecords"]
filename = ["train_record/tfrecords_mnist"]

# batch 可以自由設定
batch_size = 256

# dog and cat 共2個類別，請根據自己的資料修改
Label_size = 10

# 調用剛才的函數
image_batch, label_batch = read_and_decode(filename, batch_size)

# 轉換陣列的形狀
if is_gray:
    image_batch_train = tf.reshape(image_batch, [-1, img_width * img_height])
else:
    image_batch_train = tf.reshape(image_batch, [-1, img_width * img_height * 3])

# 把 Label 轉換成獨熱編碼
label_batch_train = tf.one_hot(label_batch, Label_size)

# W 和 b 就是我們要訓練的對象
if is_gray:
    W = tf.Variable(tf.zeros([img_width * img_height, Label_size]))
else:
    W = tf.Variable(tf.zeros([img_width * img_height * 3, Label_size]))
b = tf.Variable(tf.zeros([Label_size]))

# 我們的影像資料，會透過 x 變數來輸入
if is_gray:
    x = tf.placeholder(tf.float32, [None, img_width * img_height])
else:
    x = tf.placeholder(tf.float32, [None, img_width * img_height * 3])

# 這是參數預測的結果
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 這是每張影像的正確標籤
y_ = tf.placeholder(tf.float32, [None, Label_size])

# 計算最小交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

# 使用梯度下降法來找最佳解
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# 計算預測正確率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

####################################################
####################################################
# 新增的內容在這邊 #

# 計算 y 向量的最大值
y_pred = tf.argmax(y, 1)

# 建立 tf.train.Saver 物件
saver = tf.train.Saver()

# 將輸入與輸出值加入集合
tf.add_to_collection('input', x)
tf.add_to_collection('output', y_pred)

####################################################
####################################################

with tf.Session() as sess:
    # 初始化是必要的動作
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # 建立執行緒協調器
    coord = tf.train.Coordinator()

    # 啟動文件隊列，開始讀取文件
    threads = tf.train.start_queue_runners(coord=coord)

    # 迭代 10000 次，看看訓練的成果
    for count in range(10000):
        # 這邊開始讀取資料
        image_data, label_data = sess.run([image_batch_train, label_batch_train])

        # 送資料進去訓練
        sess.run(train_step, feed_dict={x: image_data, y_: label_data})

        # 這裡是結果展示區，每 10 次迭代後，把最新的正確率顯示出來
        if count % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: image_data, y_: label_data})
            print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy * 100))

    # 結束後記得把文件名隊列關掉
    coord.request_stop()
    coord.join(threads)

    ####################################################
    # 這裡也是新增的內容 #

    # 存檔路徑 #
    # save_path = './model/test_model'
    save_path = './model/test_model_mnist'

    # 把整張計算圖存檔
    spath = saver.save(sess, save_path)
    print("Model saved in file: %s" % spath)
    ####################################################
