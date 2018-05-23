#coding:utf-8
import tensorflow as tf
import os
from PIL import Image
import numpy as np

cwd = './dog/'
classes = {'husky', 'chihuahua'}
filename = 'dog_train.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for index, name in enumerate(classes):
    class_path = cwd+name+'/'
    for image_name in os.listdir(class_path):
        image_path = class_path+image_name

        img = Image.open(image_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes()
        # print(img_raw)
        #!注意：第一个是tf.train.features，有s
        example = tf.train.Example(features = tf.train.Features(feature={
            'label': _int64_feature(index),
            'image_raw': _bytes_feature(img_raw)
        }))
        # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())
        # 序列化为字符串
    writer.close()

