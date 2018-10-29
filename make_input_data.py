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

MAX_LEN = 50
SOS_ID = 1
SENT_THRESHOLD = 10


def MakeDataset():
    src_data,trg_data = prepare_data.prepare_gcn_data()
    if len(src_data) != len(trg_data):
        print ("src_data trg_data error!")
        exit(0)
    src_data_new = []
    trg_data_new = []
    for i in range(len(src_data)):
        if len(src_data[i]) >= SENT_THRESHOLD:
            src_data_new.append(src_data[i])
            trg_data_new.append([SOS_ID].extend(trg_data[i]))
    dataset = tf.data.Dataset.from_tensor_slices((src_data_new,trg_data_new))
    return dataset

#def MakeSrcTrgDataset(src_path,trg_path,batch_size):
#
#    dataset = MakeDataset()
#
#
#    def MakeTrgInput(trg_tuple):
#        trg_input = tf.concat([[SOS_ID],trg_label[:-1]],axis=0)
#
