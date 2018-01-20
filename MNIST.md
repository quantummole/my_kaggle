

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
import cv2
import pandas as pd
from skimage import io,filters,transform

from matplotlib import pyplot as plt
keep_prob = tf.placeholder(tf.float32)
mode = tf.placeholder(tf.bool)

tf.logging.set_verbosity(tf.logging.INFO)

STEPS = 3001
DATA_DIR='.'
BATCH_SIZE = 256

VOLUME_HEIGHT,VOLUME_WIDTH,VOLUME_DEPTH = 32,32,7
keep_prob = tf.placeholder(tf.float32)
mode = tf.placeholder(tf.bool)

tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(42)

gen = lambda x,y : [i for i in range(x,y)]

def weight_variable(shape):    
    initial = tf.truncated_normal(shape, stddev=0.1)    
    return tf.Variable(initial)

def bias_variable(shape) :
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2_layer(input,num_filters,shape,stride):
    y= tf.layers.conv2d(inputs = input,kernel_size=shape,
                     filters = num_filters,activation=tf.nn.relu,
                     padding='same',use_bias=True,strides=stride)
    return y

def batchnorm_layer(input) :
    return tf.layers.batch_normalization(inputs=input)

def full_layer(input, size,activation):
 return tf.layers.dense(inputs=input, units=size, activation=activation)

def max_pool(x,stride):
    return tf.layers.max_pooling2d(inputs=x, pool_size=stride, strides=stride)

SHUFFLE_FLAG = True
curr_index = 0
input_final = []
def getBatch(input,size) :
    global SHUFFLE_FLAG
    global curr_index
    global input_final,batch_size
    min_size =  min([len(input),size])
    if SHUFFLE_FLAG :
        shuffle(input)
        input_final = input
        SHUFFLE_FLAG = False
    curr_index += np.random.randint(1,BATCH_SIZE)
    curr_index = curr_index %len(input_final)
    return input_final[curr_index:curr_index+min_size]
```

The network design below is inspired from feature integration theory. The mid level features are integrated using max fusion and weighted fusion.
The features for detection are then computed over the integrated images using inception modules.


```python
def inception_module(inp,filter_list,num_filt) :
    feature_maps = []
    for x,y in zip(filter_list,num_filt) :
        feature_maps.append(conv2_layer(inp,y,x,(1,1)))
    return tf.concat(feature_maps,axis = 3)


def network(image_batch,FINAL_FILTERS) :
    conv_1 = batchnorm_layer(conv2_layer(image_batch,FINAL_FILTERS//32,(3,3),(1,1)))
    conv_2 = batchnorm_layer(conv2_layer(conv_1,FINAL_FILTERS//32,(3,3),(1,1)))
    pool_1 = batchnorm_layer(conv2_layer(conv_2,FINAL_FILTERS//16,(2,2),(2,2)))
    conv_3 = batchnorm_layer(conv2_layer(pool_1,FINAL_FILTERS//16,(5,5),(1,1)))
    conv_4 = batchnorm_layer(conv2_layer(conv_3,FINAL_FILTERS//16,(5,5),(1,1)))
    pool_2 = batchnorm_layer(conv2_layer(conv_4,FINAL_FILTERS//8,(2,2),(2,2)))
    max_integrator = tf.reduce_max(pool_2,axis=3,keep_dims=True)
    print(max_integrator)
    conv_integrator = batchnorm_layer(conv2_layer(pool_2,1,(1,1),(1,1)))
    print(conv_integrator)
    num_filt = [FINAL_FILTERS//8,FINAL_FILTERS//8,FINAL_FILTERS//8,FINAL_FILTERS//8]
    filt_sizes = [(1,1),(3,3),(5,5)] 
    module_1 = inception_module(max_integrator,filt_sizes,num_filt)
    module_2 = inception_module(conv_integrator,filt_sizes,num_filt)
    combined_conv = tf.concat([module_1,module_2],axis=3)
    final_conv = batchnorm_layer(conv2_layer(combined_conv,FINAL_FILTERS,(7,7),(7,7)))
    print(final_conv)
    fc1_flat = tf.reshape(final_conv,(-1,FINAL_FILTERS))
    output = full_layer(fc1_flat,10,tf.sigmoid)
    #norm = tf.reduce_sum(output_unnormalized+0.000000001)
    #output = output_unnormalized/norm 
    return output



if __name__ == '__main__' :
    train_data = pd.read_csv('./train.csv')
    train_Y = np.matrix(train_data['label']).T
    train_X = np.array([x.reshape((28,28)) for x in np.matrix(train_data.apply(lambda z : z[1:],axis = 1))])
    num_samples,m,n = train_X.shape
    train_X.shape =(num_samples,m,n,1) 
    train_indexes = [i for i in range(0,num_samples)]
    image_batch = tf.placeholder(tf.float32,name = "input",shape=(None,28,28,1))
    output_batch = tf.placeholder(tf.int32,name = "input",shape=(None,1))
    eta= tf.placeholder(tf.float32,shape=())
    output_onehot1 = tf.one_hot(output_batch,depth=10)
    output_onehot = tf.reshape(output_onehot1,shape=(-1,10))
    ones = tf.ones_like(output_onehot)
    print(output_onehot)
    output = network(image_batch,128)
    print(output)
    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=output_onehot, logits=output))

    with tf.name_scope('summaries'):
        tf.summary.scalar('cross_entropy', cross_entropy)
    summary = tf.summary.merge_all()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    label = tf.argmax(output,axis=1)
    test_data = pd.read_csv('./test.csv')
    test_X = np.array([x.reshape((28,28)) for x in np.matrix(test_data.apply(lambda z : z,axis = 1))])
    num_samples,m,n = test_X.shape
    test_X.shape =(num_samples,m,n,1)


    LOG_DIR="/home/vsl4/Mani/tensorflow/MNIST/"
    saver = tf.train.Saver(max_to_keep=0)

    with tf.Session() as sess:
            writer = tf.summary.FileWriter('{}/graphs'.format(LOG_DIR),sess.graph)
            sess.run(tf.global_variables_initializer())
            for i in range(0,STEPS) :
                train_data_indexes = getBatch(train_indexes,BATCH_SIZE)
                train_batch_X = train_X[train_data_indexes]
                train_batch_Y = train_Y[train_data_indexes]
                [lo,summ,_] = sess.run([cross_entropy,summary,train_step],feed_dict={image_batch:train_batch_X,output_batch:train_batch_Y,eta:9})
                writer.add_summary(summ,i)
                print(i,lo)
                if i%100 == 0 :
                    saver.save(sess, "{}/model_mnist.ckpt-{}".format(LOG_DIR,i))
    
    labels = []
    with tf.Session() as sess :
        saver.restore(sess,"{}model_mnist.ckpt-{}".format(LOG_DIR,2800))
        for i in range(0,num_samples,100) :
            test_batch_X = test_X[i:i+100]
            out =   sess.run(label,feed_dict={image_batch:test_batch_X})
            labels = labels + list(out)
    print(len(labels))
    ids = [i for i in range(1,len(labels)+1)]
    z = np.array([ids,labels]).T
    t = pd.DataFrame(data=z,columns = ["ImageId","Label"])
    t.to_csv("./output.csv",index=False)
```
