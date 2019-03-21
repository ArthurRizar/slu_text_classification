#coding:utf-8
###################################################################
# File Name: setting.py
# Author: Meng Zhao
# mail: @
# Created Time: Wed 21 Mar 2018 04:50:40 PM CST
#=============================================================
import os
import logging
import logging.handlers
import tensorflow as tf

#version
VERSION = '0.91'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = BASE_DIR + '/data'
MODEL_DIR = BASE_DIR + '/example/runs/v' + VERSION


#SEGMENT
SEGMENT_APP_ID = 'zsovspqm'
SEGMENT_URL= 'http://47.97.108.232:20003/term'
#SEGMENT_URL= 'http://172.16.159.177:20001/term'   #正式环境分词
SEGMENT_APP_SECRET = '4b1abe63deb7ee1117c8e386e7b16fae'



#bert path
INIT_CHECKPOINT = '/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12/bert_model.ckpt'
VOCAB_FILE = '/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12/vocab.txt'
BERT_CONFIG_FILE = '/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12/bert_config.json'




#files path
STOP_WORDS_FILE = DATA_DIR + '/stopword_data/stop_words'
#STOP_WORDS_FILE = DATA_DIR + '/stop_symbol'
LABEL_FILE = MODEL_DIR + '/labels.txt'
LABEL_MAP_FILE = MODEL_DIR + '/label_map'
CODE_FILE = MODEL_DIR + '/labelcode'



LOG_DIR = BASE_DIR + '/log/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(lineno)s]%(filename)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
log_handler = logging.handlers.TimedRotatingFileHandler(filename=LOG_DIR+'qa_intent.log', when='D', interval=1, backupCount=10)
log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger('').addHandler(log_handler)
