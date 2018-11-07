#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: prepare_for_GCN.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import codecs
import networkx as nx
import numpy as np
###
# input: full_text
###

def statis_node_property(simi_matrix):
    pass

def select_node_seq(simi_matrix,stride=1,width=10):
    degree = np.sum(simi_matrix,axis=1)
    indx_degr = zip(range(len(degree)),degree)
    indx_degr = sorted(indx_degr,key=lambda x:x[1],reverse=True)
    node_w,_ = zip(*indx_degr)
    node_chosen = [node_w[i] for i in range(0,len(node_w),stride)]
    return node_chosen

def assembly_neighbour(simi_matrix,target_node,filed_size=5):
    degree = np.sum(simi_matrix,axis=1)
    indx_degr = dict(zip(range(len(degree)),degree))
    queue_1 = []
    queue_2 = []
    queue_1.append(target_node)
    flag = [0]*len(degree)
    flag[target_node] = 1
    neighbour_list = []
    while len(neighbour_list) < filed_size:
        for v in queue_1:
            for i in range(len(degree)):
                if simi_matrix[v][i] != 0 and flag[i] == 0:
                    queue_2.append(i)
                    flag[i] = 1
        if len(queue_2) == 0:
            continue
        node = sort_node(indx_degr,queue_2)
        neighbour_list.extend(node)
        queue_1 = []
        queue_1.extend(node)
        queue_2 = []
    return neighbour_list[:filed_size]


def sort_node(indx_degr,queue_2):
    degr_node = []
    for v in queue_2:
        degr_node.append((indx_degr[v],v))
    degr_node = sorted(degr_node,key=lambda x:x[0],reverse=True)
    _,node = zip(*degr_node)
    return node

def find_neighbour(simi_matrix,center_node):
    pass
