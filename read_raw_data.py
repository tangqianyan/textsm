#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: process_rawdata.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import codecs
import re



indx_file_path = "../raw_data/cnn_stories/cnn_stories.index"
data_file_path = "../raw_data/cnn_stories/stories/"



def read_data():

    full_text = []
    full_hightlight = []
    indx1 = 0
    with codecs.open(indx_file_path,'r') as fid:
        for indx_line in fid:
            if indx1 > 100:
                break
            indx1 += 1
            indx_data_path = data_file_path + indx_line.strip()
            with codecs.open(indx_data_path,'r') as f:
                indx = 0
                data_text = []
                highlight_text = []
                highlight_flag = False
                for data_line in f:
                    if data_line == '\n':
                        continue
                    if indx == 0:
                        data_line = data_line.strip().split("--")[-1:]
                        data_line = ' '.join(data_line)
                        data_text.append(data_line)
                        indx += 1
                    else:
                        if data_line.startswith("@highlight"):
                            highlight_flag = True
                        elif highlight_flag:
                            highlight_text.append(data_line)
                            highlight_flag = False
                        else:
                            data_text.append(data_line.strip())
                full_text.append(data_text)
                full_hightlight.append(highlight_text)
    print ("full text length:  " + str(len(full_text)))

    return full_text,full_hightlight


if __name__=="__main__":
    full_text,full_hightlight = read_data(indx_file_path,data_file_path)
    build_simi_matrix(full_text)
