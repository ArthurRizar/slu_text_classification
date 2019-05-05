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
import requests
import datetime
import codecs
import multiprocessing
import numpy as np



sys.path.append('../')

from preprocess import bert_data_utils
from preprocess import datahelper
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


def request_model_predict(url, input_ids, input_mask, segment_ids, is_training, signature_name):
    data = {'inputs': {'input_ids': input_ids,
                       'input_mask': input_mask,
                       'segment_ids': segment_ids,
                       'is_training': is_training,
                     },
            'signature_name': signature_name}
    json_data = json.dumps(data)

    #print(json_data)
    #results = requests.post(url, data=json_data)
    results = requests.post(url, data=json_data)
    #print(results.text)
    result_json = json.loads(results.text)
    #print('outputs' not in result_json)
    #print(result_json)
    if 'outputs' not in result_json:
        raise Exception(str(result_json))
    pred_labels = result_json['outputs']['pred_labels']
    probs = result_json['outputs']['probs']
    return np.array(pred_labels), np.array(probs)

class Evaluator(object):
    def __init__(self, config):
        self.top_k_num = config['top_k']
        self.model_dir = config['model_dir']
        self.max_seq_length = config['max_seq_length']
        self.vocab_file = config['vocab_file']
        self.label_map_file = config['label_map_file']
        self.code_file = config['code_file']
        self.url = config['tf_serving_url']
        self.signature_name = config['signature_name']

        #init label dict and processors
        label2idx, idx2label = bert_data_utils.read_label_map_file(self.label_map_file)
        self.idx2label = idx2label
        self.label2idx = label2idx
        
        self.label2code = bert_data_utils.read_code_file(self.code_file)

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

    
        #init stop set
        self.stop_set = datahelper.get_stop_words_set(STOP_WORDS_FILE)

        #use default graph
   
    def close_session(self):
        self.sess.close()

    def evaluate(self, text):
        input_ids, input_mask, segment_ids = self.trans_text2ids(text)
       
        #print(datetime.datetime.now())
        cur_pred_labels, cur_probabilities = request_model_predict(self.url, input_ids, input_mask, segment_ids, False, self.signature_name)
        #print(datetime.datetime.now())
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
        seq_length = self.max_seq_length
        feature = bert_data_utils.convert_single_example(1, example, self.label2idx,
                                                seq_length, self.tokenizer)
        input_ids = [feature.input_ids]
        input_mask = [feature.input_mask]
        segment_ids = [feature.segment_ids]
        #print(input_ids)
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

def read_file(file_name):
    texts = []
    with codecs.open(file_name, 'r', 'utf8') as fr:
        for item in fr:
            item = item.strip()
            item_info = item.split('\t')
            text = item_info[0]
            label = item_info[1]
            texts.append(text)
    return texts

def do_request(texts, test):
    print(datetime.datetime.now())
    for text in texts:
        trunk = test.evaluate(text)
        res = get_result_json(trunk)
        #print(res)
        #exit()
    print(datetime.datetime.now())

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    config = {}
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 64
    config['top_k'] = 3
    config['code_file'] = CODE_FILE
    config['label_map_file'] = LABEL_MAP_FILE 
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['tf_serving_url'] = 'http://localhost:7831/v1/models/default:predict'
    config['signature_name'] = 'predict_text'

    test = Evaluator(config)
    #test.evaluate('我要请假')
    #print(datetime.datetime.now())
   
    texts = read_file('test.tsv')

    do_request(texts, test)

    #print(datetime.datetime.now())
