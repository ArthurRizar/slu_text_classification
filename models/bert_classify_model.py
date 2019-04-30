#coding:utf-8
###################################################
# File Name: bert_fine_tuning.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年04月29日 星期一 15时32分30秒
#=============================================================

import os
import sys
import tensorflow as tf
sys.path.append('./')

from . import modeling
from . import optimization



class BertClassifyModel(object):
    def __init__(self,
                 bert_config,
                 num_labels,
                 use_one_hot_embeddings,
                 max_seq_length,
                 init_checkpoint,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps):
        self.bert_config = bert_config
        self.num_labels = num_labels
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_seq_length = max_seq_length
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
    
        self._build_model()
        pass

    def _build_model(self):
        self.define_placeholders()
        self.init_bert_model()
        self.inference()
        self.set_loss()
        self.set_train_op()

    def define_placeholders(self):
        #self.input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
        self.input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
        #self.input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        #self.segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')
        self.segment_ids = tf.placeholder(tf.int32, [None, None], name='segment_ids')
        self.label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
        self.is_training = tf.placeholder(tf.bool, None, name='is_training') 


    def init_bert_model(self): 
        self.model = modeling.BertModel(config=self.bert_config,
                               is_training=self.is_training,
                               input_ids=self.input_ids,
                               input_mask=self.input_mask,
                               token_type_ids=self.segment_ids,
                               use_one_hot_embeddings=self.use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names
                ) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                                            init_string) 
        
    def inference(self):
        output_layer = self.model.get_pooled_output()

        keep_prob = 1.0
        def do_dropout():
            return 0.9
        def do_not_dropout():
            return 1.0
        keep_prob = tf.cond(self.is_training, do_dropout, do_not_dropout)
        self.output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)

    def set_loss(self):
        hidden_size = self.output_layer.shape[-1].value
        
        output_weights = tf.get_variable(
            "output_weights", [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            logits = tf.matmul(self.output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias, name='logits')

            self.probabilities = tf.nn.softmax(logits, axis=-1, name='probs')
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(self.label_ids, depth=self.num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels*log_probs, axis=-1)

            self.loss = tf.reduce_mean(per_example_loss)
            self.pred_labels = tf.argmax(logits, 1, name='pred_labels')

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), self.pred_labels)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc") 
        
        

    def set_train_op(self): 
        #type 1, bert default train ops
        self.train_op = optimization.create_optimizer(self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, False)


        #type 2, adam
        #self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

if __name__ == '__main__':
    BertClassifyModel('',1,False,1, '', 0.5, 1, 1)
    pass


