# -*- coding:utf-8 -*-

import sys
import six
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib


sys.path.append('../')

from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn
from tensorflow.contrib import cudnn_rnn

from common.layers.embedding import Embedding

def highway(input_, num_outputs, num_layers=1, bias=-2.0, activation=tf.nn.relu, scope='Highway'):
    '''
    Highway network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate
    '''
    size = int(num_outputs)
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = activation(tf.contrib.layers.linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(tf.contrib.layers.linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1 - t) * input_
            input_ = output

    return output

def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
                "For the tensor `%s` in scope `%s`, the actual rank "
                "`%d` (shape = %s) is not equal to the expected rank `%s`" %
                (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


class TextRNN(object):
    '''
    A cnn for text classification, following by a convolution, max-pooling, full-connection and softmax
    '''
    def __init__(self, seq_length, num_classes, vocab_size, embed_size, POS_onehot_size=0, embedding_table=None, 
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
        self.clip_gradients = clip_gradients
        self.learning_rate = learning_rate

        #placeholder
        #self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')        # shape: batch_size * seq_len
        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')        # shape: batch_size * seq_len
        #self.input_POS = tf.placeholder(tf.int32, [None, None], name='input_POS')
        self.input_POS = tf.placeholder(tf.int32, [None, seq_length], name='input_POS')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')    # shape: batch_size * num_class
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')          # dropout keep probability  
        self.is_training = tf.placeholder(tf.bool, None, name='is_training') 
        
        self.input_sparse_x = tf.sparse_placeholder(tf.int32, name='input_sparse_x')
        #regularization 
        l2_loss = tf.constant(0.0)
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        '''
        #embedding layer
        with tf.device('/cpu:0'):
            if embedding_table is None:
                self.W = tf.Variable(tf.random_uniform([vocab_size, self.embed_size], -1, 1), name='W')
            else:
                self.W = tf.Variable(embedding_table, name='W')
    
            self.embedded_words = tf.nn.embedding_lookup(self.W, self.input_x)
        '''
        
        self.embedding_layer = Embedding(vocab_size, embed_size, embedding_table=embedding_table, POS_onehot_size=POS_onehot_size)
        #self.embedded_words = self.embedding_layer.get_embedded_inputs(self.input_x, self.input_POS)
        self.embedded_words = self.embedding_layer.get_embedded_inputs(self.input_x)
        #print(self.embedded_words)
        #exit()

         
        #rnn_attn_h = self.rnn_attn_layer()
        rnn_attn_h = self.rnn_multihead_attn_layer()

        features = rnn_attn_h

        projection_hidden_size = features.shape[-1].value
        #final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                    'W',
                    shape=[projection_hidden_size, num_classes],
                    initializer=tf.contrib.layers.variance_scaling_initializer())
                    #initializer=tf.random_normal_initializer(stddev=0.1))
                    #initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_classes]), name='b')
            #l2_loss += tf.nn.l2_loss(W)
            #l2_loss += tf.nn.l2_loss(b)
            #l2_loos = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.scores = tf.nn.xw_plus_b(features, W, b, name='scores')
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


    def rnn_multihead_attn_layer(self, output_dropout=True):
        def transpose_for_scores(input_tensor,
                                 batch_size,
                                 num_attention_heads,
                                 seq_length,
                                 width):
            output_tensor = tf.reshape(input_tensor,
                                        [batch_size, seq_length, num_attention_heads, width])
            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor

        hidden_size = self.embed_size
        lstm_fw_cell = rnn.GRUCell(hidden_size)
        lstm_bw_cell = rnn.GRUCell(hidden_size)
        #lstm_fw_cell = rnn.LSTMCell(hidden_size)
        #lstm_bw_cell = rnn.LSTMCell(hidden_size)
        #lstm_fw_cell = rnn.LayerNormLSTMCell(hidden_size)
        #lstm_bw_cell = rnn.LayerNormLSTMCell(hidden_size)
        #lstm_fw_cell = rnn.LayerNormBasicLSTMCell(hidden_size)
        #lstm_bw_cell = rnn.LayerNormBasicLSTMCell(hidden_size)
        #lstm_fw_cell = rnn.LSTMBlockFusedCell(hidden_size)
        #lstm_bw_cell = rnn.LSTMBlockFusedCell(hidden_size)
        #lstm_fw_cell = cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
        #lstm_bw_cell = cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
        lstm_fw_cell = cudnn_rnn.CudnnCompatibleGRUCell(hidden_size)
        lstm_bw_cell = cudnn_rnn.CudnnCompatibleGRUCell(hidden_size)
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob)


        mask = tf.ones_like(self.input_x)
        truth_lengths = tf.reduce_sum(mask, axis=-1)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, truth_lengths, dtype=tf.float32)
        #outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32)
        output_rnn = tf.concat(outputs, axis=2)

        input_shape = get_shape_list(output_rnn, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]


        attention_size = hidden_size
        #attention_size = 50
        #attention_size = hidden_size * 2
        num_attention_heads = 5
        size_per_head = int(hidden_size / num_attention_heads)
        with tf.name_scope('multihead_attention'):
            W = tf.get_variable(name='W_attn', shape=[hidden_size*2, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name='b_attn', shape=[hidden_size], initializer=tf.random_normal_initializer(stddev=0.1)) # [attn_dim,]
            v = tf.get_variable(name='u_attn', shape=[hidden_size], initializer=tf.random_normal_initializer(stddev=0.1)) # [attn_dim,]

            # u = tanh(wh)
            Wh = tf.tensordot(output_rnn, W, axes=1, name='Wh')
            u = tf.tanh(Wh)                             # [batch_size, seq_len, hidden_size]
            
            '''
            #type 1
            multihead_u = transpose_for_scores(u, batch_size, num_attention_heads, seq_length, size_per_head)  #(B, N, T, H)
            multihead_v = tf.reshape(v, [num_attention_heads, size_per_head])       # (N, H)
            multihead_v_expand = tf.expand_dims(multihead_v, 0)                     # (1, N, H)
            multihead_v_expand = tf.expand_dims(multihead_v_expand, 2)              # (1, N, 1, H)
            uv = multihead_u * multihead_v_expand
            uv = tf.reduce_sum(uv, axis=-1)
            '''
            

            
            #type 2
            v_expand = tf.expand_dims(v, 0)
            v_expand = tf.expand_dims(v_expand, 0)
            uv = u * v_expand
            uv = tf.reshape(uv, [batch_size, num_attention_heads, seq_length, size_per_head])
            uv = tf.reduce_sum(uv, axis=-1)
            
            

            if mask is not None:
                attention_mask = tf.expand_dims(mask, axis=1)
                adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
                uv += adder


            #alphas = tf.nn.softmax(uv - tf.reduce_max(uv, -1, keepdims=True))
            alphas = tf.nn.softmax(uv)

            rnn_hidden_size_per_head = int((hidden_size * 2) / num_attention_heads) 
            multihead_output_rnn = tf.reshape(output_rnn, [batch_size, num_attention_heads, seq_length, rnn_hidden_size_per_head])
            multihead_output_attn = multihead_output_rnn*tf.expand_dims(alphas, -1)    # [batch_size, hidden_size]
            output_attn = tf.reshape(multihead_output_attn, [batch_size, seq_length, num_attention_heads*rnn_hidden_size_per_head]) 
            output_attn = tf.reduce_sum(output_attn, 1)

        #add highway
        #with tf.name_scope('highway'):
        #    output_attn = highway(output_attn, output_attn.get_shape()[1], num_layers=1, bias=0, activation=tf.nn.tanh)

        if output_dropout:
            output_attn = tf.nn.dropout(output_attn, self.dropout_keep_prob) 


        return output_attn

    def rnn_attn_layer(self, use_dropout=True):
        # rnn layer
        hidden_size = self.embed_size
        #lstm_fw_cell = rnn.BasicLSTMCell(hidden_size) #forward direction cell
        #lstm_bw_cell = rnn.BasicLSTMCell(hidden_size) #backward direction cell
        lstm_fw_cell = rnn.GRUCell(hidden_size) #forward direction cell
        lstm_bw_cell = rnn.GRUCell(hidden_size) #backward direction cell
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
        #outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, self.input_mask, dtype=tf.float32)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32)
        output_rnn = tf.concat(outputs, axis=2)     #[batch_size, sequence_length, hidden_size*2]

        #attention layer
        with tf.name_scope('attention'):
            '''
            self attention:
                alpha = softmax(v * tanh(wh))
                c_i = alpha_i * h_i
            '''
            # attention mechanism
            W = tf.Variable(tf.truncated_normal([hidden_size*2, hidden_size], stddev=0.1), name="W_attn")
            b = tf.Variable(tf.random_normal([hidden_size], stddev=0.1), name="b_attn")
            v = tf.Variable(tf.random_normal([hidden_size], stddev=0.1), name="u_attn") # [attn_dim,]
            
            # u = tanh(wh)
            u = tf.tanh(tf.tensordot(output_rnn, W, axes=1, name='Wh'))   # [batch_size, seq_len, hidden_size]
            uv = tf.tensordot(u, v, axes=1, name='uv')           # [batch_size, seq_len]

            #alphas  type1
            #exps = tf.exp(uv - tf.reduce_max(uv, -1))
            #alphas = exps / (tf.reduce_sum(exps, 1))             #softmax, get alpha , [batch_size, seq_len]
            
            #alphas  type2
            alphas = tf.nn.softmax(uv - tf.reduce_max(uv, -1, keepdims=True))
            output_attn = tf.reduce_sum(output_rnn*tf.expand_dims(alphas, -1), 1)   # [batch_size, hidden_size]

        if output_dropout:
            output_attn = tf.nn.dropout(output_attn, self.dropout_keep_prob)
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
    batch_size = 8
    seq_length = 30
    dropout_keep_prob = 0.5
    textRNN = TextRNN(seq_length=seq_length, num_classes=3, vocab_size=2000, embed_size=200, l2_reg_lambda=0.1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            # input_x should be:[batch_size, num_sentences,self.seq_len]
            #input_x = np.zeros((batch_size, seq_length+i)) #num_sentences, no padding test
            input_x = np.zeros((batch_size, seq_length)) #num_sentences
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
    print('hello world')

