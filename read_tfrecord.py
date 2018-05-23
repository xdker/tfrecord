#coding:utf-8
import img2tfrecord
import tensorflow as tf

#生成一个queue队列
filename_queue = tf.train.string_input_producer([img2tfrecord.filename])
reader = tf.TFRecordReader()

#从文件中读取一个样例，返回文件名和文件
_, serialized_example = reader.read(filename_queue)
#解析样例(单个)
features = tf.parse_single_example(serialized_example,
                                   features={'label': tf.FixedLenFeature([], tf.int64),
                                             'image_raw': tf.FixedLenFeature([], tf.string)})
img = tf.decode_raw(features['image_raw'], tf.uint8)
# img = img.reshape(img, [128,128,3])
# img = img.cast(img, tf.float32)*(1/255)-0.5
label = tf.cast(features['label'], tf.int32)
print(img, label)
