
Transfer learning using VGG and fully connected layers is trained to classify dog breeds.


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:59:05 2018

@author: QuantumMole
"""

import tensorflow as tf
import numpy as np
from random import shuffle
import vgg
import pandas as pd
from skimage import io,filters,transform
from matplotlib import pyplot as plt
import os

def max_pool(x,shape,stride):
    return tf.layers.max_pooling2d(inputs=x, pool_size=shape, strides=stride)


def conv2_layer(input,num_filters,shape,stride,scale_l1 = 0.1,scale_l2=0.1):
    regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1,scale_l2)
    y= tf.layers.conv2d(inputs = input,kernel_size=shape,
                     filters = num_filters,activation=None,
                     padding='same',use_bias=True,strides=stride,kernel_regularizer = regularizer)
    return y

def full_layer(input, size,activation):
 regularizer = tf.contrib.layers.l1_l2_regularizer(0.1,1.0)
 return tf.layers.dense(inputs=input, units=size, activation=activation,kernel_regularizer = regularizer,bias_regularizer = regularizer)

def droput_layer(input,keep_prob) :
    return tf.nn.dropout(input, keep_prob=keep_prob)
def batchnorm_layer(input) :
    return tf.layers.batch_normalization(inputs=input)

def getTrainData() : 
    labels = pd.read_csv('./labels.csv')
    breeds = pd.read_csv('./breeds.csv')
    y_augmented = labels.set_index('breed').join(breeds.set_index('breed')).set_index('id')
    pictures = [x for x in os.listdir('./train') if 'jpg' in x]
    y = np.array([y_augmented.get_value(x.replace(".jpg",""),'breed_id') for x in pictures])
    #z = np.array([transform.resize(io.imread("./train/{}".format(x)),(256,256)) for x in pictures])
    return np.array(pictures),y


```


```python
pictures,Y = getTrainData()
vgg_model = vgg.VGG16()
BATCH_SIZE = 8
pictures_train = pictures[0:int(0.8*len(pictures))]
pictures_valid = pictures[int(0.8*len(pictures)):]  
train_Y = Y[0:int(0.8*len(pictures))]
valid_Y = Y[int(0.8*len(pictures)):]
pictures_test = np.array([x for x in os.listdir('./test') if 'jpg' in x])
def loadImages(image_indices,pics,fold,valid=False) :
    def preprocess(im) :
        z = np.random.randint(0,1)
        if z == 1:
            x = np.random.randint(0,223)
            y =np.random.randint(0,223)
            im[x:x+16,y:y+16] = 0           
        return im
    if not valid :
        z = np.array([preprocess(transform.resize(io.imread("./{}/{}".format(fold,x)),(128,128))) for x in pics[image_indices]])
    else :
        z = np.array([transform.resize(io.imread("./{}/{}".format(fold,x)),(128,128)) for x in pics[image_indices]])

    return z
```


```python
    LOG_DIR="/home/vsl4/Mani/tensorflow/DogBreed"
    with vgg_model.graph.as_default() :
        keep_prob = tf.placeholder(tf.float32)
        final_layer1 = tf.nn.relu(batchnorm_layer(conv2_layer(vgg_model.get_layer_tensors([12])[0],256,(1,1),(1,1),1.0,0.0)))
        final_layer2 = max_pool(tf.nn.relu(batchnorm_layer(conv2_layer(final_layer1,256,(5,5),(1,1),1.0,0.0))),(2,2),(2,2))
        final_layer = tf.nn.relu(batchnorm_layer(conv2_layer(final_layer2,512,(3,3),(1,1),0.0,1.0)))
        final_layer_flat = tf.reshape(final_layer,(-1,512*16))
        fc_1 = droput_layer(tf.nn.relu(batchnorm_layer(full_layer(final_layer_flat,1024,None))),keep_prob)
        fc_2 = droput_layer(tf.nn.relu(batchnorm_layer(full_layer(fc_1,1024,None))),keep_prob)
        output = full_layer(fc_2,120,tf.nn.relu)        
        oimage_batch = tf.placeholder(tf.int32,shape=(None,),name = "image_output")
        output_labels = tf.one_hot(oimage_batch,120)
        output_probs = tf.nn.softmax(output)
        loss = tf.losses.softmax_cross_entropy(output_labels,output)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
        with tf.name_scope('summaries'):
            tf.summary.scalar('Loss', loss)
        summary = tf.summary.merge_all()
        validation_loss = tf.summary.scalar('validation_loss', loss)

        saver = tf.train.Saver(max_to_keep=100)
        final_ans = 0.0
        with tf.Session() as sess:
            runs = [i for i in range(0,8161,480)]
            writer = tf.summary.FileWriter('{}/graphs'.format(LOG_DIR),sess.graph)
            sess.run(tf.global_variables_initializer())
            for epoch in range(0,31*len(pictures_train),len(pictures_train)) :
                for i in range(0,len(pictures_train),BATCH_SIZE) :
                    train_data_indexes = range(i,min(len(pictures_train),i+BATCH_SIZE))
                    train_batch_X = loadImages(train_data_indexes,pictures_train,'train')
                    train_batch_Y = train_Y[train_data_indexes]
                    vgg_dict = {**vgg_model.create_feed_dict(train_batch_X),oimage_batch:train_batch_Y,keep_prob:0.5}
                    _,loss1,summ= sess.run([train_step,loss,summary],feed_dict=vgg_dict)
                    writer.add_summary(summ,epoch+i)
                    valid_indices = [i for i in range(len(pictures_valid))]
                    shuffle(valid_indices)
                    valid_indices = valid_indices[0:32]
                    train_batch_X = loadImages(valid_indices,pictures_train,'train',True)
                    train_batch_Y = valid_Y[valid_indices]
                    vgg_dict = {**vgg_model.create_feed_dict(train_batch_X),oimage_batch:train_batch_Y,keep_prob:1.0}
                    val_loss= sess.run(validation_loss,feed_dict=vgg_dict)
                    writer.add_summary(val_loss,epoch+i)
                    if i%96 == 0:
                        saver.save(sess, "{}/model.ckpt-{}-{}".format(LOG_DIR,epoch,i))
                    print(epoch+i,loss1)
                indices = shuffle([i for i in range(len(pictures_train))])
                pictures_train = pictures_train[indices].reshape(-1)
                train_Y = train_Y[indices].reshape(-1)
            epoch = 30*len(pictures_train),
            for run in runs :    
                image_data = []
                saver.restore(sess,"{}/model.ckpt-{}-{}".format(LOG_DIR,epoch,run))
                for i in range(0,len(pictures_test),BATCH_SIZE) :
                    train_data_indexes = range(i,min(len(pictures_test),i+BATCH_SIZE))
                    train_batch_X = loadImages(train_data_indexes,pictures_test,'test',True)
                    vgg_dict = {**vgg_model.create_feed_dict(train_batch_X),keep_prob:1.0}
                    out = sess.run(output_probs,feed_dict=vgg_dict)
                    image_data.append(out)
                final_ans = final_ans+np.vstack(image_data)
            final_ans = final_ans/1.0/len(runs)
```


```python
filenames = np.array([x.replace(".jpg","") for x in pictures_test])
filenames.shape = (filenames.shape[0],1)
final_ans_norm = np.array([x/np.sum(x) for x in final_ans])
z = np.hstack([final_ans_norm,filenames])
breeds = pd.read_csv('./breeds.csv')
breeds.set_index('breed_id')
col_names = [breeds.get_value(i,'breed') for i in range(120)] +['id']
t = pd.DataFrame(data=z,columns = col_names)
t = t.set_index('id')
t.to_csv("./output.csv")
```
