#!/root/zhaomeng/anaconda/bin
#coding:utf-8
###################################################
# File Name: httpserver.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年06月06日 星期三 15时24分44秒
#=============================================================

import os
import sys
import json


import numpy as np
import datetime
import logging


sys.path.append('../')


from global_config import *
from evaluator import Evaluator
from common.segment.segment_client import SegClient

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='intent.log', level=logging.DEBUG, format=LOG_FORMAT)

import os
from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import wsgi
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU
a = tf.placeholder(tf.int32, shape=(), name="input")
asquare = tf.multiply(a, a, name="output")
sess = tf.Session()  # 创建tensorflow session，也可以在这里载入tensorflow模型
pred_instance = Evaluator()


settings = {}
settings['url'] = SEGMENT_URL
settings['appId'] = SEGMENT_APP_ID
settings['appSecret'] = SEGMENT_APP_SECRET
seg_client = SegClient(settings)


app = Flask(__name__)

def get_segment_str(seg_client, text):
    '''
    print text
    logging.info(text)
    segs_json = seg_client.segment_text(text)
    segs = segs_json['data']

    segs = json.loads(segs)
    rs = ''
    for item in segs:
        word = item['word']
        POS = item['nature']
        rs += word + '/' + POS + ' '
    rs = rs.strip()
    return rs.encode('utf-8')
    '''
    
    from pyhanlp import *
    rs = ''
    for term in HanLP.segment(text):
        word = str(term.word)
        POS = str(term.nature)
        rs += word + '/' + POS + ' '
    rs = rs.strip()
    return rs.encode('utf8')




@app.route('/')
def index():
    return 'Hello World'

@app.route('/predict')
def response_request():
    text = request.args.get('text')
    #print text
    #print type(text)
    text = text.encode('utf8')
    logging.info('..start..')

    text_segs = get_segment_str(seg_client, text)
    logging.info('..end segment..')
    trunk = pred_instance.evaluate(text_segs)
    logging.info('..end predict..')
    response = []
    for cur_label, cur_code, cur_score in trunk:
        #self.write(str(cur_label) + ' ' + str(cur_score) + '\n')
        intent_item = {}
        intent_item['title'] = str(cur_label).decode('utf-8')
        intent_item['name'] = str(cur_code)
        intent_item['score'] = str(cur_score)
        #json_item = json.dump(intent_itemt)
        response.append(intent_item)
    response_json = json.dumps(response, ensure_ascii=False)
    logging.info('..end..')
    return response_json

if __name__ == "__main__":
    server = wsgi.WSGIServer(('0.0.0.0', 19877), app)
    server.serve_forever()
