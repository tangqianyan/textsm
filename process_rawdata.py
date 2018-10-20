#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: process_rawdata.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import codecs

indx_file_path = "../raw_data/cnn_stories/cnn_stories.index"
data_file_path = "../raw_data/cnn_stories/stories/"

def read_data(indx_file_path,data_file_path):

    with codecs.open(indx_file_path,'r') as f:
        for line in f:
            indx_data_path = data_file_path + line.strip()
            with codecs.open(indx_data_path,'r') as f:
                for 
