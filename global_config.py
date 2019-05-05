#coding:utf-8
# File Name: config.py
#=============================================================
import os
import logging
import logging.handlers
import configparser

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


cf = configparser.ConfigParser()
cf_interpolation = configparser.ExtendedInterpolation()
cf.read(BASE_DIR+'/global_config.cfg')



#version
VERSION = '1.0'
DATA_DIR = BASE_DIR + '/data'
MODEL_DIR = BASE_DIR + '/runs/v' + VERSION


#SEGMENT
SEGMENT_URL= 'http://127.0.0.1/segment'


TF_SERVING_REST_PORT = cf['web']['rest_api_port']
TF_SERVING_CLIENT_PORT = cf['web']['client_port']




#files path
STOP_WORDS_FILE = BASE_DIR + cf['file_path']['stopword_file']
LABEL_FILE = BASE_DIR + cf['file_path']['label_file']
LABEL_MAP_FILE = BASE_DIR + cf['file_path']['label_map_file'].format(VERSION)
CODE_FILE = BASE_DIR + cf['file_path']['code_file'].format(VERSION)



LOG_DIR = BASE_DIR + '/log/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(lineno)s]%(filename)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
log_handler = logging.handlers.TimedRotatingFileHandler(filename=LOG_DIR+'qa_intent.log', when='D', interval=1, backupCount=10)
log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger('').addHandler(log_handler)
