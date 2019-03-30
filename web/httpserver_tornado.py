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
import tornado.web
import tornado.ioloop


import numpy as np
import datetime
import logging


sys.path.append('../')


from setting import *
from evaluator import Evaluator
from common.segment.segment_client import SegClient

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='intent.log', level=logging.DEBUG, format=LOG_FORMAT)


os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU

class PredictHttpServer(tornado.web.RequestHandler):
    def initialize(self, seg_client):
        pred_instance = Evaluator()
        self.pred_instance = pred_instance
        self.seg_client = seg_client

    def get_segment_str(self, text):
        print text
        logging.info(text)
        segs_json = self.seg_client.segment_text(text)
        segs = segs_json['data']
        
        segs = json.loads(segs)
        rs = ''
        for item in segs:
            word = item['word']
            POS = item['nature']
            rs += word + '/' + POS + ' '
        rs = rs.strip()
        return rs.encode('utf-8')


    def get(self):
        text_uni = self.get_argument('text')
        text = text_uni.encode('utf-8')
        print datetime.datetime.now(), 'begin.....'
        text_segs = self.get_segment_str(text)
        print datetime.datetime.now(), 'segment.....'

        trunk = self.pred_instance.evaluate(text_segs)
        print datetime.datetime.now(), 'predict.....'
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
        print 'end............'
        self.write(response_json)


    def post(self):
        text_uni = self.get_body_argument('text')
        text = text_uni.encode('utf-8')
        text_segs = self.get_segment_str(text)

        trunk = self.pred_instance.evaluate(text_segs)
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
        self.write(response_json)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print '!! nedd two paramert (port and thread num)\n'
        exit(0)
    port = int(sys.argv[1])
    num_processes = int(sys.argv[2])

    settings = {}
    settings['url'] = SEGMENT_URL
    settings['appId'] = SEGMENT_APP_ID
    settings['appSecret'] = SEGMENT_APP_SECRET
    seg_client = SegClient(settings)
        

    #pred_instance = Evaluator()
    #pred_instance.evaluate('测试/v')
    application = tornado.web.Application([
            (r"/predict", PredictHttpServer, 
            dict(
            seg_client=seg_client
            )),
        ])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.bind(port)
    http_server.start(num_processes)
    tornado.ioloop.IOLoop.instance().start()
