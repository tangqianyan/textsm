#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: build_simi_matrix.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import re
import numpy as np
#import process_raw_data




def build_simi_matrix(document):

    drop = "a in on the an to of from at as no there do if not by or is are am were was have has can \
            may could might be for and with nor that which this ".split()

    indx = 0
    word_set = []
    sent_list = []
    doc_len = len(document)
    simi_matrix = np.zeros((doc_len,doc_len))
    for sentence in document:
        sentence = sentence.lower()
        sentence = re.sub("[--\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！?，。？、~@#￥%……&*（）]+',","",sentence)
        sent_list.append(sentence.split())

    for i in range(doc_len):
        for j in range(i+1,doc_len):
            num = len((set(sent_list[i])&set(sent_list[j]))-set(drop))
            if j == i+1:
                num = 1
            num = 1 if num > 0 else 0
            simi_matrix[i][j] = num
            simi_matrix[j][i] = num

    num_list = np.sum(simi_matrix,axis=1)
    #for i in range(len(num_list)):
    #    print str(i) + ":" + str(i*2+1) + ": " + str(num_list[i])

    np.savetxt('ss.txt',simi_matrix,fmt='%d')
    return simi_matrix
