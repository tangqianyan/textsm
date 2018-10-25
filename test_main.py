#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: test_main.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import process_rawdata
import build_simi_matrix
import prepare_for_GCN
import get_sent_vec
import cPickle
import random

indx_file_path = "../raw_data/cnn_stories/cnn_stories.index"
data_file_path = "../raw_data/cnn_stories/stories/"


def main():

    full_text,full_hightlight = process_rawdata.read_data(indx_file_path,data_file_path)
    simi_matrix = build_simi_matrix.build_simi_matrix(full_text)

    full_embed = get_sent_vec.get_sent_vec(full_text)
    cPickle.dump([full_embed,full_hightlight],open("../data/cnn_stories/cnn_stories_sent_highlight_vec_50.p","wb"))
    #print full_embed[0]

    #print simi_matrix
    #ne = prepare_for_GCN.assembly_neighbour(simi_matrix,1,10)
    #print ne


if __name__=="__main__":
    main()
