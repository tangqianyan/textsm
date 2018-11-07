#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: process_rawdata.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import read_raw_data
import collections
import codecs
from operator import itemgetter

VOCAB_SIZE = 100000

vocab_path = "../data/cnn_stories/cnn_stories.vocab"

def process_raw_data():
    print("process_raw_data ...")

    full_text,full_highlight = read_raw_data.read_data()
    sorted_word,full_text_new,full_highlight_new = build_word_dict(full_text,full_highlight)
    full_highlight_num = convert_to_number(sorted_word,full_text_new,full_highlight_new)
    return full_text,full_highlight_num


def clear_raw_text(sent):
    sent = sent.lower()
    return sent

def build_word_dict(full_text,full_highlight):
    print("build_word_dict ...")

    word_dict = collections.Counter()
    full_text_new = []
    full_highlight_new = []
    for doc in full_text:
        for sent in doc:
            sent = clear_raw_text(sent)
            #full_text_new.append(sent)
            for v in sent:
                word_dict[v] += 1
    for doc in full_highlight:
        highlight_list = []
        for sent in doc:
            sent = clear_raw_text(sent)
            highlight_list.append(sent)
            for v in sent:
                word_dict[v] += 1
        full_highlight_new.append(highlight_list)

    sorted_word = sorted(word_dict.items(),
                        key =lambda x:x[1],
                        reverse=True)
    sorted_word = [v[0] for v in sorted_word]
    sorted_word = ["<eos>","<unk>","<sos>"] + sorted_word
    sorted_word = sorted_word[:VOCAB_SIZE]

    with codecs.open(vocab_path,'w')  as f:
        for v in sorted_word:
            f.write(v+'\n')
    return sorted_word,full_text_new,full_highlight_new

def convert_to_number(sorted_word,full_text_new,full_highlight_new):
    print ("convert_to_number ...")
    #full_text_num = []
    full_highlight_num = []
    word_dict = dict(zip(sorted_word,range(len(sorted_word))))
    indx = 0
    for doc in full_highlight_new:
        if indx%1000 == 0:
            print indx
        indx += 1
        doc_list = []
        for sent in doc:
            sent_new = []
            for v in sent:
                sent_new.append(word_dict[v])
            doc_list.append(sent_new)
        full_highlight_num.append(doc_list)
    return full_highlight_num


if __name__=="__main__":
    process_raw_data()

