#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: make_input.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import tensorflow as tf
import re
import codecs
import process_rawdata
import build_simi_matrix


indx_file_path = "../raw_data/cnn_stories/cnn_stories.index"
data_file_path = "../raw_data/cnn_stories/stories/"

def read_data():
    full_text,full_hightlight = process_rawdata.read_data(indx_file_path,data_file_path)
    build_simi_matrix.build_simi_matrix(full_text)



if __name__=="__main__":
    read_data()

