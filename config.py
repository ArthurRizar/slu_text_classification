#coding:utf-8
# File Name: config.py
#=============================================================
import os
import logging
import logging.handlers
import tensorflow as tf

#version
VERSION = '1.0'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = BASE_DIR + '/data'
MODEL_DIR = BASE_DIR + '/runs/v' + VERSION


#SEGMENT
SEGMENT_URL= 'http://127.0.0.1/segment'






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
