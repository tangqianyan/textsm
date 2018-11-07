#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: prepare_data.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import tensorflow as tf

import process_rawdata
import build_simi_matrix
import prepare_for_GCN
import get_sent_vec
import cPickle
import pickle
import random

indx_file_path = "../raw_data/cnn_stories/cnn_stories.index"
data_file_path = "../raw_data/cnn_stories/stories/"
sent_file_path = "../data/cnn_stories/cnn_stories_sent_highlight_vec_50.p"

def prepare_gcn_data():
    print("prepare data ...")

    #full_text,full_hightlight_num = process_rawdata.process_raw_data(indx_file_path,data_file_path)
    full_text,full_hightlight = process_rawdata.process_raw_data()

    with open(sent_file_path,'rb') as f:
        sent_vec_list = pickle.load(f)[0]

    data_final = []
    hightlight_final = []
    indx = 0
    for i in range(len(full_text)):
        #if i == 49:
        #    print full_text[i]
        #print "document: " + str(i)
        document = full_text[i]
        if len(document) < 10:
            continue
        indx += 1
        if indx > 5:
            break
        doc_sent_vec = sent_vec_list[i]
        simi_matrix = build_simi_matrix.build_simi_matrix(document)
        nodes_selected = prepare_for_GCN.select_node_seq(simi_matrix,1,10)
        for node in nodes_selected:
            neighbors = []
            data_doc = []
            receptive_field = []
            neighbors = prepare_for_GCN.assembly_neighbour(simi_matrix,node,5)
            neighbors = [node] + neighbors
            for v in neighbors:
                receptive_field.append(doc_sent_vec[v])
            data_doc.append(receptive_field)
        data_final.append(data_doc)
        hightlight_final.append(full_hightlight[i])
    return data_final,hightlight_final

#if __name__=="__main__":
#    main()
