#coding:utf-8
###################################################
# File Name: text_rnn.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年03月27日 星期二 17时11分59秒
#=============================================================
#TextRNN: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat output, 4.FC layer, 5.softmax
import sys
import numpy as np
import tensorflow as tf

sys.path.append('../')
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



class TextRNN:
    def __init__(self, seq_len, num_classes, vocab_size, embed_size, POS_onehot_size, decay_steps=1000, decay_rate=0.9, use_subword=False,
                 embedding_table=None, l2_reg_lambda=0.0, learning_rate=0.001, initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        #self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.l2_reg_lambda = l2_reg_lambda
        self.embedding_table = embedding_table
        self.num_sampled = 20
        self.POS_onehot_size = POS_onehot_size


        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x")  # X
        self.input_POS = tf.placeholder(tf.int32, [None, self.seq_len], name='input_POS')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")  # y [None,num_classes]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')

        self.input_sparse_x = tf.sparse_placeholder(tf.int32, name='input_sparse_x')


        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits, self.predictions = self.inference() #[None, self.label_size]. main computation graph is here.
        self.loss = self.get_loss() #-->self.get_loss_nce()
        self.train_op = self.train()

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.predictions, tf.argmax(self.input_y, 1)) #tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.embedding_layer = Embedding(self.vocab_size, self.embed_size, self.POS_onehot_size, embedding_table=self.embedding_table)
            self.embed_size += self.POS_onehot_size
            self.hidden_size = self.embed_size


            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size*2, self.num_classes], initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])       #[label_size]
            


    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        #1.get emebedding of words in the sentence
        self.embedded_words = self.embedding_layer.get_embedded_inputs(self.input_x, self.input_POS) #shape:[None,sentence_length, embed_size]


        #2. Bi-lstm layer
        # define lstm cess:get lstm cell output
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size) #forward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size) #backward direction cell
        #lstm_fw_cell = rnn.LayerNormBasicLSTMCell(self.hidden_size) #forward direction cell
        #lstm_bw_cell = rnn.LayerNormBasicLSTMCell(self.hidden_size) #backward direction cell
        if self.dropout_keep_prob is not None:
            #lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            #lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                            where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32) #[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        print("outputs:===>", outputs) #outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))

        # 3. concat 
        output_rnn = tf.concat(outputs, axis=2) #[batch_size,sequence_length,hidden_size*2]
        

        attention_dim = 100
        with tf.name_scope('attention'):
            '''
            self attention:
                alpha = softmax(v * tanh(wh))
                c_i = alpha_i * h_i
            '''
            attn_hidden_size = output_rnn.shape[2].value
            # attention mechanism
            W = tf.Variable(tf.truncated_normal([attn_hidden_size, attention_dim],stddev=0.1), name="W_attn")
            b = tf.Variable(tf.random_normal([attention_dim], stddev=0.1), name="b_attn")

            v = tf.Variable(tf.random_normal([attention_dim], stddev=0.1), name="u_attn") # [attn_dim,]
            u = tf.tanh(tf.matmul(tf.reshape(output_rnn, [-1, attn_hidden_size]), W) + tf.reshape(b, [1, -1])) #[batch_size, seq_len, attn_dim] 

            uv = tf.matmul(u, tf.reshape(v, [-1, 1]))

            exps = tf.reshape(tf.exp(uv), [-1, self.seq_len])
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1]) #softmax, get alpha
            output_attn = tf.reduce_sum(output_rnn * tf.reshape(alphas, [-1, self.seq_len, 1]), 1)



        self.output_rnn_last = output_attn
        #self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1) #[batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] #TODO
        print("output_rnn_last:", self.output_rnn_last) # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>

        #batch normalization
        #self.output_rnn_last = tf.layers.batch_normalization(self.output_rnn_last, training=self.is_training)
      
        '''
        #add highway
        with tf.name_scope('highway'):
            self.h_highway = highway(self.output_rnn_last, self.output_rnn_last.get_shape()[1], 1, 0, tf.nn.tanh, 'highway')
            self.output_rnn_last = self.h_highway
        '''

        #add dropout 
        with tf.name_scope('dropout'):
            self.output_rnn_last = tf.nn.dropout(self.output_rnn_last, self.dropout_keep_prob)

        #4. logits(use linear layer)
        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            #logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
            
            predictions = tf.argmax(logits, axis=1, name="predictions") #shape:[None,]
        return logits, predictions

    def get_loss(self,l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss = tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def get_loss_nce(self,l2_lambda=0.0001): #0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
            
        #labels = tf.reshape(self.input_y,[-1])               #[batch_size,1]------>[batch_size,]
        labels = tf.expand_dims(tf.argmax(self.input_y, 1), 1)                   #[batch_size,]----->[batch_size,1]
        loss = tf.reduce_mean( #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                tf.nn.nce_loss(weights=tf.transpose(self.W_projection),#[hidden_size*2, num_classes]--->[num_classes,hidden_size*2]. nce_weights:A `Tensor` of shape `[num_classes, dim].O.K.
                               biases=self.b_projection,                 #[label_size]. nce_biases:A `Tensor` of shape `[num_classes]`.
                               labels=labels,                 #[batch_size,1]. train_labels, # A `Tensor` of type `int64` and shape `[batch_size,num_true]`. The target classes.
                               inputs=self.output_rnn_last,# [batch_size,hidden_size*2] #A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                               num_sampled=self.num_sampled,  #scalar. 100
                               num_classes=self.num_classes,partition_strategy="div"))  #scalar. 1999
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

#test started
def test():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=10
    learning_rate=0.01
    decay_steps=1000
    decay_rate=0.9
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1#0.5
    textRNN=TextRNN(sequence_length, num_classes, vocab_size, embed_size, learning_rate, decay_steps, decay_rate, is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
            input_y=input_y=np.array([1,0,1,1,1,2,1,1]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss,acc,predict,_=sess.run([textRNN.loss,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
            print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
#test()
