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

from tornado.web import RequestHandler, Application, asynchronous

from multiprocessing import Pool, TimeoutError

from setting import *
from evaluator_new import Evaluator
from preprocess import dataloader
from common.segment.segment_client import SegClient


os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU

stop_set = dataloader.get_stop_words_set(STOP_WORDS_FILE)



class PredictHttpServer(tornado.web.RequestHandler):
    #def __init__(self, application, request, **kwargs):
    #    super(tornado.web.RequestHandler, self).__init__()



    def initialize(self, eval_pool):
        self.eval_pool = eval_pool


    def have_need_arguments(self):
        if ('text'   not in self.request.arguments) or \
           ('method' not in self.request.arguments):
            return False
        else:
            return True

    def get_intents_table(self):
        table = []
        idx2label = self.pred_instance.idx2label
        label2code = self.pred_instance.label2code

        for idx in idx2label:
            intent_dict = {}
            intent_dict['id'] = int(idx) + 1
            
            label = idx2label[idx]
            intent_dict['title'] = label
            intent_dict['name'] = label2code[label]
            table.append(intent_dict)

        table_json = json.dumps(table, ensure_ascii=False)
        return table_json

    def http_predict(self, text_uni):

        #logging.warning('end char:'+str(text_uni[-1].encode('utf8')))
        #logging.warning(str(text_uni[-1].encode('utf8') in stop_set))
        #if text_uni[-1].encode('utf8') in stop_set:
        #    text_uni = text_uni[:-1]
        msg = '' 
        response = []
        try:
            trunk = self.eval_pool.apply_async(func=evaluator.do_evaluate_task, args=(text_uni,))
            #trunk = self.eval_pool.apply(func=evaluator.do_evaluate_task, args=(text_uni,))
            trunk = trunk.get()
            for cur_label, cur_code, cur_score in trunk:
                #self.write(str(cur_label) + ' ' + str(cur_score) + '\n')
                intent_item = {}
                intent_item['title'] = cur_label
                intent_item['name'] = cur_code
                intent_item['score'] = str(cur_score)
                #json_item = json.dump(intent_itemt)
                response.append(intent_item)
        except Exception as err:
            msg = str(err)
        return response, msg   
 
    @tornado.web.asynchronous
    def get(self):
        err_dict = {}
        
        #判断请求参数是否存在
        if 'method' not in self.request.arguments:
            err_dict['errMsg'] = '缺少请求参数method'
            self.write(json.dumps(err_dict, ensure_ascii=False))
            return
        
        #判断method_type类型
        method_type = self.get_argument('method')
        if method_type != 'getSupportIntentions' and method_type != 'predict':
            err_dict['errMsg'] = '参数method错误'
            self.write(json.dumps(err_dict, ensure_ascii=False))
            return
        elif method_type == 'getSupportIntentions':
            table_json = self.get_intents_table()
            self.write(table_json)
            return

        if 'text' not in self.request.arguments:
            err_dict['errMsg'] = '缺少请求参数text'
            self.write(json.dumps(err_dict, ensure_ascii=False))
            return 


        text_uni = self.get_argument('text')
        response, msg = self.http_predict(text_uni)
        
        #若predict时出错
        if msg != '':
            err_dict['errMsg'] = msg
            self.write(json.dumps(err_dict, ensure_ascii=False))
            return

        response_json = json.dumps(response, ensure_ascii=False)
        self.write(response_json)
        self.finish()

    @tornado.web.asynchronous
    def post(self):
        err_dict = {}

        #判断请求参数是否存在
        if 'method' not in self.request.arguments:
            err_dict['errMsg'] = '缺少请求参数method'
            self.write(json.dumps(err_dict, ensure_ascii=False))
            return

        #判断method_type类型
        method_type = self.get_body_argument('method')
        if method_type != 'getSupportIntentions' and method_type != 'predict':
            err_dict['errMsg'] = '参数method错误'
            self.write(json.dumps(err_dict, ensure_ascii=False))
            return
        elif method_type == 'getSupportIntentions':
            table_json = self.get_intents_table()
            self.write(table_json)
            return

        if 'text' not in self.request.arguments:
            err_dict['errMsg'] = '缺少请求参数text'
            self.write(json.dumps(err_dict, ensure_ascii=False))
            return

        text_uni = self.get_body_argument('text')
        response, msg = self.http_predict(text_uni)

        #若predict时出错
        if msg != '':
            err_dict['errMsg'] = msg
            self.write(json.dumps(err_dict, ensure_ascii=False))
            return

        response_json = json.dumps(response, ensure_ascii=False)
        self.write(response_json)
        self.finish()

def init_evaluator(config):
    global cur_evaluator
    evaluator = Evaluator(config)

def do_evaluate_task(text):
    trunk = cur_evaluator.evaluate(text)
    return trunk


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
            self.write('index')


if __name__ == '__main__':


    config = {}
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 32
    config['top_k'] = 3
    config['code_file'] = CODE_FILE
    config['label_map_file'] = LABEL_MAP_FILE
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'


    eval_pool = Pool(processes=3, initializer=evaluator.init_evaluator, initargs=(config,))
    

    application = tornado.web.Application([
            (r"/intent", PredictHttpServer, 
            dict(eval_pool=eval_pool,)
            ),
        ])
    

    #application.listen(7732)

    server = tornado.httpserver.HTTPServer(application)
    server.bind(7732)
    server.start(1)

    tornado.ioloop.IOLoop.instance().start()
