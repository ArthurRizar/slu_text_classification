# -*- coding:utf-8 -*-

import sys
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib


sys.path.append('../')

from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn

from setting import *
from common.layers.embedding import Embedding

def highway(input_, num_outputs, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    '''
    Highway network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate
    '''
    size = int(num_outputs)
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(tf.contrib.layers.linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(tf.contrib.layers.linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1 - t) * input_
            input_ = output

    return output



class TextRNN(object):
    '''
    A cnn for text classification, following by a convolution, max-pooling, full-connection and softmax
    '''
    def __init__(self, seq_len, num_classes, vocab_size, embed_size, filter_sizes, num_filters, embedding_table=None, 
            l2_reg_lambda=0.0, decay_steps=1000, decay_rate=0.9, clip_gradients=5.0, learning_rate=1e-4):
        '''
        @brief:
        @param: seq_length, sequence length
                num_classes, num of classes
                vocab_size, size of word vocab  
                embe_size, size of embedding
                filter_sizes, size of each filter , a list, like [3, 4, 5]
                num_filters, num of filter
                l2_reg_lambda, scale of l2 regularization

        '''
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding_table = embedding_table
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.clip_gradients = clip_gradients
        self.learning_rate = learning_rate

        #placeholder
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')        # shape: batch_size * seq_len
        self.input_POS = tf.placeholder(tf.int32, [None, None], name='input_POS')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')    # shape: batch_size * num_class
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')          # dropout keep probability  
        self.is_training = tf.placeholder(tf.bool, None, name='is_training') 
        
        self.input_sparse_x = tf.sparse_placeholder(tf.int32, name='input_sparse_x')
        #regularization 
        l2_loss = tf.constant(0.0)
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        #embedding layer
        with tf.device('/cpu:0'):
            if embedding_table is None:
                self.W = tf.Variable(tf.random_uniform([vocab_size, self.embed_size], -1, 1), name='W')
            else:
                self.W = tf.Variable(embedding_table, name='W')
    
            self.embedded_words = tf.nn.embedding_lookup(self.W, self.input_x)

            self.embedded_word_expand = tf.expand_dims(self.embedded_words, -1)
         
        self.h_flat = self.rnn_attn_layer()

        '''
        #add highway
        with tf.name_scope('highway'):
            self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0, tf.nn.tanh)
            self.h_pool_flat = self.h_highway
        '''

        #add dropout 
        with tf.name_scope('dropout'):
            #self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)
            self.h_drop = tf.nn.dropout(self.h_flat, self.dropout_keep_prob)


        #final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            num_filters_total = sum(self.num_filters)
            W = tf.get_variable(
                    'W',
                    shape=[self.embed_size*2, num_classes],
                    initializer=tf.contrib.layers.variance_scaling_initializer())
                    #initializer=tf.random_normal_initializer(stddev=0.1))
                    #initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            #l2_loss += tf.nn.l2_loss(W)
            #l2_loss += tf.nn.l2_loss(b)
            #l2_loos = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        #calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #losses = tf.reduce_mean(tf.square(self.scores - self.input_y))  #mse
            #losses = -tf.reduce_mean(self.input_y * tf.log(tf.clip_by_value(tf.nn.softmax(self.scores), 1e-10, 1.0)))  
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        #accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


        self.train_op = self.get_train_op()



    def rnn_attn_layer(self):
        # rnn layer
        self.hidden_size = self.embed_size
        #lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size) #forward direction cell
        #lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size) #backward direction cell
        lstm_fw_cell = rnn.GRUCell(self.hidden_size) #forward direction cell
        lstm_bw_cell = rnn.GRUCell(self.hidden_size) #backward direction cell
        #lstm_fw_cell = rnn.LayerNormBasicLSTMCell(self.hidden_size) #forward direction cell
        #lstm_bw_cell = rnn.LayerNormBasicLSTMCell(self.hidden_size) #backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            #lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            #lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                            where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, self.embedded_words.shape[1].value, dtype=tf.float32)
        output_rnn = tf.concat(outputs, axis=2)     #[batch_size, sequence_length, hidden_size*2]

        #attention layer
        attention_dim = self.embed_size
        with tf.name_scope('attention'):
            '''
            self attention:
                alpha = softmax(v * tanh(wh))
                c_i = alpha_i * h_i
            '''
            attn_hidden_size = output_rnn.shape[2].value

            # attention mechanism
            W = tf.Variable(tf.truncated_normal([attn_hidden_size, attention_dim], stddev=0.1), name="W_attn")
            b = tf.Variable(tf.random_normal([attention_dim], stddev=0.1), name="b_attn")

            v = tf.Variable(tf.random_normal([attention_dim], stddev=0.1), name="u_attn") # [attn_dim,]
            v_expand = tf.expand_dims(v, -1)
            
            # u = tanh(wh)
            u = tf.tanh(tf.tensordot(output_rnn, W, axes=1, name='Wh'))   # [batch_size, seq_len, attention_dim]

            uv = tf.tensordot(u, v, axes=1, name='uv')           # [batch_size, seq_len]

            #alphas  type1
            #exps = tf.exp(uv - tf.reduce_max(uv, -1))
            #alphas = exps / (tf.reduce_sum(exps, 1))             #softmax, get alpha , [batch_size, seq_len]
            
            #alphas  type2
            alphas = tf.nn.softmax(uv - tf.reduce_max(uv, -1, keep_dims=True))
            
            output_attn = tf.reduce_sum(output_rnn*tf.expand_dims(alphas, -1), 1)   # [batch_size, attn_hidden_size]
            
            print output_attn

        return output_attn

    def get_train_op(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op





if __name__ == '__main__':
    textRNN = TextRNN(seq_len=20, num_classes=3, vocab_size=2000, embed_size=200, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100], l2_reg_lambda=0.1)
    with tf.Session() as sess:
        batch_size = 8
        seq_len = 20
        dropout_keep_prob = 0.5
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            # input_x should be:[batch_size, num_sentences,self.seq_len]
            input_x = np.zeros((batch_size, seq_len+i)) #num_sentences
            input_x[input_x > 0.5] = 1
            input_x[input_x <= 0.5] = 0
            print(np.shape(input_x))
            input_y = np.matrix(
                [[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])  # np.zeros((batch_size),dtype=np.int32) #[None, self.seq_len]
            loss, acc, predict, _ = sess.run(
                [textRNN.loss, textRNN.accuracy, textRNN.predictions, textRNN.train_op],
                feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y,
                           textRNN.dropout_keep_prob: dropout_keep_prob})
            print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)
    print 'hello world'

