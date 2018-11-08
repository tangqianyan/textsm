#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: gcnt.py
#Author: chi xiao
#Mail: 
#Created Time:
############################
import tensorflow as tf
import numpy as np
import make_input_data
import gcn

CHECKPOINT_PATH="../model/seq2seq_ckpt"
NUM_LAYERS = 1
HIDDEN_SIZE = 87
BATCH_SIZE = 1
NUM_EPOCH = 1
WORD_VOCAB_SIZE = 40000
MAX_GRAD_NORM = 5
class GCNTSUM(object):

    def __init__(self):
        self.enc_cell = gcn
        #self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
        #    [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        #    for _ in range(NUM_LAYERS)])

        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
            for _ in range(NUM_LAYERS)])
        self.word_embedding = tf.get_variable(
            "word_emb",
            [WORD_VOCAB_SIZE,HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb",
            [WORD_VOCAB_SIZE,HIDDEN_SIZE])

        self.softmax_weight = tf.get_variable(
            "weight",[HIDDEN_SIZE,WORD_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "bias",[WORD_VOCAB_SIZE])

    def forward(self,src_input,src_size,trg_input,trg_label,trg_size=0):
        batch_size = tf.shape(src_input)[0]

        #src_emb = tf.nn.embedding_lookup(self.word_embedding,src_input)
        trg_emb = tf.nn.embedding_lookup(self.word_embedding,trg_input)

        with tf.variable_scope("encoder"):
            enc_outputs = gcn.inference(src_input,src_size,0)
            enc_outputs = tf.squeeze(enc_outputs)
            enc_outputs = [enc_outputs]
            enc_outputs = tf.convert_to_tensor(enc_outputs)
            #enc_outputs = self.dec_cell.zero_state(1,np.float32)
            #enc_outputs = tf.cast(enc_outputs,tf.float32)
            #enc_outputs = (enc_outputs)
            #enc_outputs = tf.reshape(enc_outputs,(batch_size,50))
            #enc_outputs = tf.convert_to_tensor(np.zeros((BATCH_SIZE,50)))
            #enc_outputs = (enc_outputs,enc_outputs)

        with tf.variable_scope("decoder"):
            h0 = self.dec_cell.zero_state(1,np.float32)
            #h0 = enc_outputs
            #h0 = tf.convert_to_tensor(np.ones((1,50)))
            #h0 = tf.cast(h0,tf.float32)
            #h0 = (h0,)
            #h0 = tf.tuple([h0])
            print h0
            print trg_emb
            print trg_emb.shape
            #trg_emb = tf.reshape(trg_emb,shape=(1,shape1[0],shape1[1]))
            #trg_emb = tf.convert_to_tensor([trg_emb])
            dec_outputs,_ = tf.nn.dynamic_rnn(self.dec_cell,
                                             trg_emb,
                                             trg_size,
                                             initial_state=h0)

        output = tf.reshape(dec_outputs,[-1,HIDDEN_SIZE])
        logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label,[-1]),
                                                              logits=logits)

        label_weights = tf.sequence_mask(trg_size,
                                        #maxlen=tf.shape(trg_label)[1],
                                         maxlen=50,
                                        dtype=tf.float32)

        label_weights = tf.reshape(label_weights,[-1])
        cost = tf.reduce_sum(loss*label_weights)
        #cost = tf.reduce_sum(loss)
        cost_per_token = cost / tf.reduce_sum(label_weights)
        #cost_per_token = cost

        trainable_variables = tf.trainable_variables()

        grads = tf.gradients(cost / tf.to_float(batch_size),
                            trainable_variables)

        grads,_ = tf.clip_by_global_norm(grads,MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads,trainable_variables))
        return cost_per_token,train_op,h0

def run_epoch(session,cost_op,train_op,saver,step):
    while True:
        try:
            cost,_ = session.run([cost_op,train_op])
            if step % 10 ==0:
                print("After %d steps, per token cost is %0.3f" % (step,cost))
            if step % 200 == 0:
                saver.save(session,CHECKPOINT_PATH,global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    initializer = tf.random_uniform_initializer(-0.05,0.05)

    with tf.variable_scope("GCNTSUM_model",
                           reuse=None,
                           initializer=initializer):
        train_model = GCNTSUM()

    #data = make_input_data.MakeDataset()
    #iterator = data.make_initializable_iterator()
    #(src,src_size),(trg_input,trg_label,trg_size) = iterator.get_next()
    #src,trg_input,trg_label,trg_size = iterator.get_next()
    src_size = 0
    #src = np.array([src])
    #src = tf.convert_to_tensor(src)
    #print (src)
    #trg_input = np.array([trg_input])
    #trg_label = np.array([trg_label])
    #print(src.shape)
    #print(trg_input.shape)
    #print(trg_label.shape)
    #src = tf.placeholder(np.float32,shape=(1,6,50))
    #trg_input = tf.placeholder(np.int32,shape=(1,50))
    #trg_label = tf.placeholder(np.int32,shape=(1,50))
    if True:
        data = make_input_data.MakeDataset()
        iterator = data.make_initializable_iterator()
        src,trg_input,trg_label,trg_size = iterator.get_next()
        trg_input = tf.squeeze(trg_input)
        trg_input = tf.reshape(trg_input,(1,87))
        trg_label = tf.squeeze(trg_label)
        trg_label = tf.reshape(trg_label,(1,87))
        trg_size = np.array([87])
        #trg_label = tf.reshape(trg_label,(1,trg_label.shape[0]))
    else:
        src = tf.convert_to_tensor(np.zeros((1,6,50)))
        trg_input = tf.convert_to_tensor(np.ones((1,50)))
        trg_input = tf.cast(trg_input,tf.int32)
        trg_label = tf.convert_to_tensor(np.ones((1,50)))
        trg_label = tf.cast(trg_label,tf.int32)
        trg_size = np.array([50])
    cost_op, train_op,h0 = train_model.forward(src,
                                           src_size,
                                           trg_input,
                                           trg_label,
                                           trg_size)
    saver = tf.train.Saver()
    step = 0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print ("In iteration: %d" %(i+1))
            #sess.run(iterator.initializer)
            #print h0.eval()
            #print trg_input.eval()
            #step = run_epoch(sess,
            #                cost_op,
            #                train_op,
            #                saver,
            #                step)
if __name__ == "__main__":
    main()
