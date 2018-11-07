#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: make_input_data.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import tensorflow as tf
import prepare_data
import numpy as np

MAX_LEN = 50
EOS_ID = 0
SOS_ID = 2
SENT_THRESHOLD = 0


def MakeDataset():
    print ("make dataset ...")

    src_data,trg_data = prepare_data.prepare_gcn_data()
    print ("src_data size: ") + str(np.asarray(src_data).shape)
    print ("trg_data size: ") + str(len(trg_data))
    if len(src_data) != len(trg_data):
        print ("src_data trg_data error!")
        exit(0)
    src_data_new = []
    trg_input_data_new = []
    trg_label_data_new = []
    trg_size = []
    for i in range(len(src_data)):
        if len(src_data[i]) >= SENT_THRESHOLD:
            src_data_new.append(src_data[i])
            trg_input_data_new.append([SOS_ID]+ trg_data[i][0])
            trg_label_data_new.append(trg_data[i][0] + [EOS_ID])
            trg_size.append(len(trg_data[i]))
 

    src_data_new = np.array(src_data_new)
    #print src_data_new.shape
    trg_input_data_new = pad_sentence(trg_input_data_new)
    trg_input_data_new = np.array(trg_input_data_new)
    #print trg_input_data_new.shape
    trg_label_data_new = pad_sentence(trg_label_data_new)
    trg_label_data_new = np.array(trg_label_data_new)
    #print trg_label_data_new.shape
    trg_size = np.array(trg_size)
    #dataset = tf.contrib.data.Dataset.from_tensor_slices(src_data_new)
    #dataset = tf.contrib.data.Dataset.from_tensor_slices(trg_input_data_new)
    #dataset = tf.contrib.data.Dataset.zip((src_data_new,trg_input_data_new,trg_label_data_new,trg_size))
    dataset = tf.contrib.data.Dataset.from_tensor_slices((src_data_new,trg_input_data_new,trg_label_data_new,trg_size))
    padded_shape = (
        (tf.TensorShape([None])),
        (tf.TensorShape([None])),
        (tf.TensorShape([None])),
        (tf.TensorShape([]))
    )
    #ba_data = dataset.padded_batch(1,padded_shape)
    print("dataset")
    return dataset

def pad_sentence(sentences,padding_word_id=1):
    maxlen = max(len(x) for x in sentences)
    padded_sent = []
    for v in sentences:
        num_padd = maxlen - len(v)
        new_sent = v + [padding_word_id]*num_padd
        padded_sent.append(new_sent)
    return padded_sent

#def MakeSrcTrgDataset(src_path,trg_path,batch_size):
#
#    dataset = MakeDataset()
#
#
#    def MakeTrgInput(trg_tuple):
#        trg_input = tf.concat([[SOS_ID],trg_label[:-1]],axis=0)
#
