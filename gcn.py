#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: gcn.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import tensorflow as tf
import numpy as np
import prepare_data

filed_size = 5
seq_size = 1

INPUT_NODE = 0
OUTPUT_NODE = 0
IMAGE_SIZE = 0
NUM_CHANNELS =1 
CONV1_H = filed_size + 1
CONV1_W = 1
CONV1_D = 1
CONV2_H = seq_size
CONV2_W = 1
CONV2_D = 1

def inference(input_tensor,train,regularizer):

    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight",
            [CONV1_H,CONV1_W,NUM_CHANNELS,CONV1_D],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            "bias",
            [CONV1_D],
            initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,
                            conv1_weights,
                            strides=[1,filed_size+1,1,1],
                            padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable(
            "weight",
            [CONV2_W,CONV2_H,CONV1_D,CONV2_D],
            initializer=tf.truncated_normal_initializer(0.1))
        conv2_biases = tf.get_variable(
            "bias",
            [CONV2_D],
            initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1,
                            conv2_weights,
                            strides=[1,1,1,1],
                            padding="VALID")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    return relu2

if __name__=="__main__":
    num = np.ones((2,6,4,1))
    num = tf.convert_to_tensor(num,tf.float32)
    m = inference(num,0,0)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print sess.run(m)

