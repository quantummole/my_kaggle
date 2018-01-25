
Deep features are extracted from the last convolutional layer of VGG net, the features are integrated by max and mean fusion.
The resulting vector is then used to train 2 Random Forests- one for max integrated features and the other for mean integrated features.
The final probabilities are then estimated by finding the number of tress that vote for a given class.


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:59:05 2018

@author: vsl4
"""

import tensorflow as tf
import numpy as np
from random import shuffle
import vgg
import pandas as pd
from skimage import io,filters,transform
from matplotlib import pyplot as plt
import os


def getTrainData() : 
    labels = pd.read_csv('./labels.csv')
    breeds = pd.read_csv('./breeds.csv')
    y_augmented = labels.set_index('breed').join(breeds.set_index('breed')).set_index('id')
    pictures = [x for x in os.listdir('./train') if 'jpg' in x]
    y = np.array([y_augmented.get_value(x.replace(".jpg",""),'breed_id') for x in pictures])
    #z = np.array([transform.resize(io.imread("./train/{}".format(x)),(256,256)) for x in pictures])
    return np.array(pictures),y[:,np.newaxis]
```


```python
pictures,train_Y = getTrainData()
vgg_model = vgg.VGG16()
BATCH_SIZE = 64
def loadImages(image_indices,pics,fold) :
    z = np.array([transform.resize(io.imread("./{}/{}".format(fold,x)),(256,256)) for x in pics[image_indices]])
    return z
```


```python
image_data_max = []
image_data_mean = []
with vgg_model.graph.as_default() :
    final_layer = vgg_model.get_layer_tensors([-1])[0]
    max_fusion = tf.reduce_max(final_layer,axis=3)
    mean_fusion = tf.reduce_mean(final_layer,axis=3)
    output_max = tf.reshape(max_fusion,(-1,256))
    output_mean = tf.reshape(mean_fusion,(-1,256))
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(0,len(pictures),BATCH_SIZE) :
                train_data_indexes = range(i,min(len(pictures),i+BATCH_SIZE))
                train_batch_X = loadImages(train_data_indexes,pictures,'train')
                train_batch_Y = train_Y[train_data_indexes]
                vgg_dict = vgg_model.create_feed_dict(train_batch_X)
                [out_max,out_mean] = sess.run([output_max,output_mean],feed_dict=vgg_dict)
                image_data_max.append(out_max)
                image_data_mean.append(out_mean)
                print(i)

train_data = np.vstack(image_data_max)
np.save('X_train_max.npy',train_data)
train_data = np.vstack(image_data_mean)
np.save('X_train_mean.npy',train_data)

image_data_max = []
image_data_mean = []
pictures_test = np.array([x for x in os.listdir('./test') if 'jpg' in x])
with vgg_model.graph.as_default() :
    final_layer = vgg_model.get_layer_tensors([-1])[0]
    max_fusion = tf.reduce_max(final_layer,axis=3)
    mean_fusion = tf.reduce_mean(final_layer,axis=3)
    output_max = tf.reshape(max_fusion,(-1,256))
    output_mean = tf.reshape(mean_fusion,(-1,256))
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(0,len(pictures_test),BATCH_SIZE) :
                train_data_indexes = range(i,min(len(pictures_test),i+BATCH_SIZE))
                train_batch_X = loadImages(train_data_indexes,pictures_test,'test')
                vgg_dict = vgg_model.create_feed_dict(train_batch_X)
                [out_max,out_mean] = sess.run([output_max,output_mean],feed_dict=vgg_dict)
                image_data_max.append(out_max)
                image_data_mean.append(out_mean)
                print(i)
test_data = np.vstack(image_data_max)
np.save('X_test_max.npy',test_data)
test_data = np.vstack(image_data_mean)
np.save('X_test_mean.npy',test_data)
```


```python
from sklearn.ensemble  import RandomForestClassifier
tx_max = np.load('X_train_max.npy')
tx_mean = np.load('X_train_mean.npy')
clf_max = RandomForestClassifier(max_depth=10, random_state=0,n_estimators=100)
clf_max.fit(tx_max,train_Y)
clf_mean = RandomForestClassifier(max_depth=10, random_state=0,n_estimators=100)
clf_mean.fit(tx_mean,train_Y)
testx_max = np.load('X_test_max.npy')
testx_mean = np.load('X_test_mean.npy')
final_ans_max = clf_max.predict_proba(testx_max)
final_ans_mean = clf_mean.predict_proba(testx_mean)
final_ans = np.array(final_ans_max) + np.array(final_ans_mean)
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
