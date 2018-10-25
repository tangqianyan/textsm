#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: get_sent_vec.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import numpy as np


glove_file_path = "../word2vec_data/glove.6B.50d.txt"


def loadGlove():
    vocab = []
    embed = []
    vocab.append("<unk>")
    embed.append([0]*50)
    with open(glove_file_path,'r') as f:
        for line in f:
            line = line.strip().split()
            vocab.append(line[0])
            embed.append(map(float,line[1:]))
    print "load Glove embeddings ..."
    print "vocab size : " + str(len(vocab))
    print "embedding dimension : " +str(len(embed[0]))
    embed = np.asarray(embed)

    return vocab,embed

def get_sent_vec(full_text):
    vocab,embed = loadGlove()
    vocab_embed = dict(zip(vocab,embed))
    full_embed = []
    indx = 0
    for doc in full_text:
        indx += 1
        if indx%1000 == 0:
            print indx
        vec_list = []
        for sent in doc:
            vec = vocab_embed["<unk>"]
            for v in sent:
                if v in vocab_embed:
                    vec += vocab_embed[v]
            vec = vec / len(sent)
            vec_list.append(vec)
        full_embed.append(vec_list)
        #break
    return full_embed


