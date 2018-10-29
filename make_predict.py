#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: make_predict.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import tensorflow as tf

CHECKPOINT_PATH = "../model/"

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
SHARE_EMB_AND_SOFTMAX = True

SOS_ID = 1
EOS_ID = 2


class TGCNModel(object):

    def __init__(self):
        self.enc_cell = gcn()
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)]
            for _ in range(NUM_LAYERS))

    def inference(self,src_input):

        with tf.variable_scope("encoder"):
            enc_outputs = gcnt()

    MAX_DEC_LENGTH = 100
    with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
        init_array = tf.TensorArray(dtype=tf.int32,
                                   size = 0,
                                   dynamic_size=True,
                                   clear_after_read=False)
        init_array = init_array.write(0,SOS_ID)
        init_loopvar = (enc_outputs,init_array,0)

        def continue_loop_condition(state,trg_ids,step):
            return tf.reduce_all(tf.logical_and(
                tf.not_equal(trg_ids.read(step),EOS_ID),
                tf.less(step,MAX_DEC_LENGTH-1)
            ))

        def loop_body(state,trg_ids,step):
            trg_input = [trg_ids.read(step)]
            trg_emb = tf.nn.embedding_lookup(self.trg_embedding ,trg_input)
            dec_outputs, next_state = self.dec_cell.call(
                state=state,inputs=trg_emb)
            output = tf.reshape(dec_outputs,[-1,HIDDEN_SIZE])
            logits = (tf.matmul(output,self.softmax_weight)+self.softmax_bias)
            next_id = tf.argmax(logits,axis=1,output_type=tf.int32)
            trg_ids = trg_ids.write(step+1,next_id[0])
            return next_state,trg_ids,step+1

        state, trg_ids, step = tf.while_loop(continue_loop_condition,loop_body,init_loopvar)
        return trg_ids.stack()


def main():

    with tf.variable_scope("TGCN_model",reuse=None):
            model = TGCNModel()
    test_data = [""]
    output_op = model.inference(test_data)
    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess,CHECKPOINT_PATH)
    output = sess.run(output_op)
    print (output)
    sess.close()

if __name__=="__main__":
    main()
