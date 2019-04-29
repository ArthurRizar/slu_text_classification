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


import evaluator

#from tornado.concurrent import run_on_executor
#from concurrent.futures import ThreadPoolExecutor


from multiprocessing import Pool, TimeoutError

from setting import *
from evaluator import Evaluator
from preprocess import dataloader
from common.segment.segment_client import SegClient


os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU

stop_set = dataloader.get_stop_words_set(STOP_WORDS_FILE)



class PredictHttpServer(tornado.web.RequestHandler):
    #def __init__(self, application, request, **kwargs):
    #    super(tornado.web.RequestHandler, self).__init__()


    def initialize(self, pred_instance):
        self.pred_instance = pred_instance


    def check_arguments(self):
        if 'text' not in self.request.arguments:
            raise Exception('text is required')
        if 'method' not in self.request.arguments:
            raise Exception('method is required (predict or getSupportIntentions)')
        
        req_method = self.get_argument('method')
        #print(req_method)
        if req_method not in ['predict', 'getSupportIntentions']:
            raise Exception("method must be 'predict' or 'getSupportIntentions'")

    def get_intents_table(self):
        table = []
        idx2label = self.pred_instance.idx2label
        label2code = self.pred_instance.label2code

        for idx in idx2label:
            intent_dict = {}
            intent_dict['id'] = int(idx) + 1
            
            label = idx2label[idx]
            intent_dict['title'] = label
            intent_dict['name'] = label2code[label.lower()]
            table.append(intent_dict)

        table_json = json.dumps(table, ensure_ascii=False)
        return table_json

    def http_predict(self):
        try:
            text_uni = self.get_argument('text')
            logging.info(text_uni)
            response = []
            trunk = self.pred_instance.evaluate(text_uni)
            for cur_label, cur_code, cur_score in trunk:
                intent_item = {}
                intent_item['title'] = cur_label
                intent_item['name'] = cur_code
                intent_item['score'] = str(cur_score)
                #json_item = json.dump(intent_itemt)
                response.append(intent_item)
        except Exception as err:
            raise err
        return response 
 
    def get(self):
        err_dict = {}
        try:
            self.check_arguments()     
            method_type = self.get_argument('method')
            if method_type == 'predict':
                response = self.http_predict()
            else:
                response = self.get_intents_table()
            response_json = json.dumps(response, ensure_ascii=False)
            self.write(response_json)

        except Exception as err:
            err_dict['errMsg'] = repr(err)
            self.write(json.dumps(err_dict, ensure_ascii=False))
            logging.warning(repr(err))
        

    def post(self):
        err_dict = {}
        try:
            self.check_arguments()
            method_type = self.get_argument('method')
            if method_type == 'predict':
                response = self.http_predict()
            else:
                response = self.get_intents_table()
            response_json = json.dumps(response, ensure_ascii=False)
            self.write(response_json)

        except Exception as err:
            err_dict['errMsg'] = repr(err)
            self.write(json.dumps(err_dict, ensure_ascii=False))
            logging.warning(repr(err))


if __name__ == '__main__':


    config = {}
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 32
    config['top_k'] = 3
    config['code_file'] = CODE_FILE
    config['label_map_file'] = LABEL_MAP_FILE
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'

    pred_instance = Evaluator(config)
    pred_instance.evaluate('测试')


    application = tornado.web.Application([
            (r"/intent", PredictHttpServer, 
            dict(pred_instance=pred_instance,)
            ),
        ])
    application.listen(13965)
   
    #server = tornado.httpserver.HTTPServer(application)
    #server.bind(7732)
    #server.start(1)

    tornado.ioloop.IOLoop.instance().start()
