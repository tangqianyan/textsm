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

class GCNTSUM(object):

    def __init__(self):
        self.enc_cell = gcn()
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)]
            for _ in range(NUM_LAYERS))

        self.word_embedding = tf.get_variable(
            "word_emb",
            [WORD_VOCAB_SIZE,HIDDEN_SIZE])

    def forward(self,src_input,src_size,trg_input,trg_label,trg_size):
        batch_size = tf.shape(src_input)[0]

        src_emb = tf.nn.embedding_lookup(self.word_embedding,src_input)
        trg_emb = tf.nn.embedding_lookup(self.word_embedding,trg_input)

        with tf.variable_scope("encoder"):
            enc_outputs = gcn(src_emb,src_size,dtype=tf.float32)

        with tf.variable_scope("decoder"):
            dec_outputs,_ = tf.nn.dynamic_rnn(self.dec_cell,
                                             trg_emb,
                                             trg_size,
                                             initial_state=enc_outputs)

        output = tf.reshape(dec_outputs,[-1,HIDDEN_SIZE])
        logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label,[-1]),
                                                              logits=logits)

        label_weights = tf.sequence_mask(trg_size,
                                        maxlen=tf.shape(trg_label)[1],
                                        dtype=tf.float32)

        label_weights = tf.reshape(label_weights,[-1])
        cost = tf.reduce_sum(loss*label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        trainable_variables = tf.trainable_variables()

        grads = tf.gradients(cost / tf.to_float(batch_size),
                            trainable_variables)

        grads,_ = tf.clip_by_global_norm(grads,MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads,trainable_variables))
        return cost_per_token,train_op

def run_epoch(session,cost_op,train_op,saver,step):
    while True:
        try:
            cost,_ = session.run({cost_op,train_op})
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

    data = make_input_data.MakeDataset()
    iterator = data.make_initializable_iterator()
    (src,src_size),(trg_input,trg_label,trg_size) = iterator.get_next()
    cost_op, train_op = train_model.forward(src,
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
            sess.run(iterator.initializer)
            step = run_epoch(sess,
                            cost_op,
                            train_op,
                            saver,
                            step)
if __name__ == "__main__":
    main()
