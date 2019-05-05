#coding:utf-8
###################################################
# File Name: eval.py
# Author: Meng Zhao
# mail: @
# Created Time: Fri 23 Mar 2018 09:27:09 AM CST
#=============================================================
import os
import sys
import csv
import json
import logging
import numpy as np


sys.path.append('../')

from preprocess import bert_data_utils
from preprocess import dataloader
from preprocess import tokenization
from global_config import *
from tensorflow.contrib import learn


os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='intent.log', level=logging.DEBUG, format=LOG_FORMAT)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def norm(x):
    x[0] = x[0] * 10
    return x / x.sum()


class Evaluator(object):
    def __init__(self, config):
        self.top_k_num = config['top_k']
        self.model_dir = config['model_dir']
        self.max_seq_length = config['max_seq_length']
        self.vocab_file = config['vocab_file']
        self.label_map_file = config['label_map_file']
        self.code_file = config['code_file']
        self.model_checkpoints_dir = config['model_checkpoints_dir']
        self.model_pb_path = config['model_pb_path']


        #init label dict and processors
        label2idx, idx2label = bert_data_utils.read_label_map_file(self.label_map_file)
        self.idx2label = idx2label
        self.label2idx = label2idx
        
        self.label2code = bert_data_utils.read_code_file(self.code_file)

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

    
        #init stop set
        self.stop_set = dataloader.get_stop_words_set(STOP_WORDS_FILE)

        #use default graph
        self.graph = tf.get_default_graph()
        restore_graph_def = tf.GraphDef()
        restore_graph_def.ParseFromString(open(self.model_pb_path, 'rb').read())
        tf.import_graph_def(restore_graph_def, name='')

        session_conf = tf.ConfigProto()
        self.sess = tf.Session(config=session_conf)
        self.sess.as_default()
        self.sess.run(tf.global_variables_initializer())

        #restore model
        #cp_file = tf.train.latest_checkpoint(self.model_checkpoints_dir)
        #saver = tf.train.import_meta_graph('{}.meta'.format(cp_file))
        #saver.restore(self.sess, cp_file)

        #get the placeholders from graph by name
        self.input_ids = self.graph.get_operation_by_name('input_ids').outputs[0]
        self.input_mask = self.graph.get_operation_by_name('input_mask').outputs[0]
        self.segment_ids = self.graph.get_operation_by_name('segment_ids').outputs[0]
        self.is_training = self.graph.get_operation_by_name('is_training').outputs[0]

        #tensors we want to evaluate
        self.pred_labels = self.graph.get_operation_by_name('loss/pred_labels').outputs[0]
        self.probabilities = self.graph.get_operation_by_name('loss/probs').outputs[0]
        self.logits = self.graph.get_operation_by_name('loss/logits').outputs[0]
   
    def close_session(self):
        self.sess.close()

    def evaluate(self, text):
        input_ids, input_mask, segment_ids = self.trans_text2ids(text)

        feed_dict = {
                self.input_ids: input_ids,
                self.input_mask: input_mask,
                self.segment_ids: segment_ids,
                self.is_training: False}
        cur_pred_labels, cur_probabilities, cur_logits = self.sess.run([self.pred_labels, self.probabilities, self.logits], feed_dict)
        #print(cur_pred_labels)
        #print(cur_probabilities)
        
        best_label = self.idx2label[cur_pred_labels[0]]
        
        all_ids = np.argsort(-cur_probabilities, 1)
        top_k_ids = all_ids[:, :self.top_k_num][0]

        top_k_labels = [self.idx2label[idx] for idx in top_k_ids]
        top_k_probs = cur_probabilities[0][top_k_ids]
        top_k_probs = norm(top_k_probs)
        top_k_code = [self.label2code[label] for label in top_k_labels]

        return zip(top_k_labels, top_k_code, top_k_probs) 


    def trans_text2ids(self, text):
        example = bert_data_utils.InputExample(guid='1', text_a=text)
        length = len(text) + 2
        feature = bert_data_utils.convert_single_example(1, example, self.label2idx,
                                                length, self.tokenizer)
        input_ids = [feature.input_ids]
        input_mask = [feature.input_mask]
        segment_ids = [feature.segment_ids]
        return input_ids, input_mask, segment_ids 

def get_result_json(trunk):
    response = []
    for cur_label, cur_code, cur_score in trunk:
        #self.write(str(cur_label) + ' ' + str(cur_score) + '\n')
        intent_item = {}
        intent_item['title'] = cur_label
        intent_item['name'] = cur_code
        intent_item['score'] = str(cur_score)
        #json_item = json.dump(intent_itemt)
        response.append(intent_item) 
    res = json.dumps(response, ensure_ascii=False)
    return res



def init_evaluator(config):
    global cur_evaluator
    cur_evaluator = Evaluator(config)

def do_evaluate_task_test(i):
    trunk = cur_evaluator.evaluate('公司的班车什么时候到呢？')
    return trunk

def do_evaluate_task(text):
    trunk = cur_evaluator.evaluate(text)
    return trunk

if __name__ == '__main__':
    config = {}
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 128
    config['top_k'] = 3
    config['code_file'] = CODE_FILE
    config['label_map_file'] = LABEL_MAP_FILE 
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'

    '''
    test = Evaluator(config)
    trunk = test.evaluate('公司班车')
    response = []
    for cur_label, cur_code, cur_score in trunk:
        #self.write(str(cur_label) + ' ' + str(cur_score) + '\n')
        intent_item = {}
        intent_item['title'] = cur_label
        intent_item['name'] = cur_code
        intent_item['score'] = str(cur_score)
        #json_item = json.dump(intent_itemt)
        response.append(intent_item)
    print(json.dumps(response, ensure_ascii=False))
    test.close_session()
    '''

    from multiprocessing import Pool, TimeoutError
    import datetime

    pool = Pool(processes=1, initializer=init_evaluator, initargs=(config,))
    for i in range(100):
        #pool.apply_async(func=task, args=(i,))
        trunk = pool.apply(func=do_evaluate_task_test, args=(i,))
        res = pool.apply_async(os.getpid, ())
        #print(res.get())
        #print(get_result_json(trunk))

    print(datetime.datetime.now())
    for i in range(100):
        #pool.apply_async(func=task, args=(i,))
        trunk = pool.apply(func=do_evaluate_task_test, args=(i,))
        res = pool.apply(os.getpid, ())
    print(datetime.datetime.now())


