#coding:utf-8
###################################################
# File Name: connect_UNIT.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年05月14日 星期一 17时54分49秒
#=============================================================
import urllib
import urllib2
import json
import time
import httplib
import datetime
import hashlib
import requests
import numpy as np


def http_get_request():
    #url = 'http://192.168.7.233:19877/predict?'
    url = 'http://192.168.7.233:7732/predict?'
    post_data = {}

    with open('query.txt', 'r') as fr:
        for line in fr:
            line = line.strip()
            cur_url = url + 'text=' + urllib.quote(line)
            #print cur_url
            request = urllib2.Request(cur_url)
            request.add_header('Content-Type', 'application/json')
            response = urllib2.urlopen(request)
            content = response.read()
            if (content):
            #print content 
                rs = json.loads(content)
                #print rs


def http_post_request():
    #url = 'http://47.96.228.29:8892/predict'
    #url = 'http://192.168.7.233:19877/predict'
    url = 'http://192.168.7.233:13965/qa_intent'
    post_data = {}
    with open('query.txt', 'r') as fr:
        for line in fr:
            line = line.strip().lower()
            #print line
            line_info = line.split('\t')
            #post_data['text'] = line
            post_data['text'] = line_info[0]
            #post_data['method'] = 'getSupportIntentions'
            post_data['method'] = 'predict'
            post_data_urlencode = urllib.urlencode(post_data)
            request = urllib2.Request(url, post_data_urlencode)
            #request.add_header('Content-Type', 'application/json')
            response = urllib2.urlopen(request)
            content = response.read()
            if (content):
                rs = json.loads(content)
                print(line_info[0]+'\t'+rs[0]['title'].encode('utf8'))
                #exit()

def http_post_request_test():
    #url = 'http://47.96.228.29:8892/predict'
    #url = 'http://192.168.7.233:19877/predict'
    url = 'http://192.168.7.233:7732/encode'
    post_data = {}
    with open('query.txt', 'r') as fr:
        for line in fr:
            line = line.strip().lower()
            #print line
            line_info = line.split('\t')
            #post_data['text'] = line
            post_data['id'] = 123
            post_data['texts'] = [line_info[0]]
            post_data['is_tokenized'] = False
            request = urllib2.Request(url, data=json.dumps(post_data))
            request.add_header('Content-Type', 'application/json')
            response = urllib2.urlopen(request)
            content = response.read()
            if (content):
            #print content 
                rs = json.loads(content)
                #print(line_info[0]+'\t'+rs[0]['title'].encode('utf8'))




def get_sign(post_data, app_secret):
    items = sorted(post_data.items(), key=lambda x:x[0])
    print items
    items_str = ''
    for key, value in items:
        items_str += str(key) + str(value)
    print items_str
    
    items_str = app_secret + items_str + app_secret

    md5_handler = hashlib.md5()
    md5_handler.update(items_str)
    rs = md5_handler.hexdigest()
    print rs
    return rs

def segment_test(words):
    url = 'http://47.97.108.232:20003/term'
    post_data = {}
    post_data['words'] = words
    appSecret_str = '4b1abe63deb7ee1117c8e386e7b16fae'
    post_data['appId'] = 'zsovspqm'
    post_data['timestamp'] = str(int(time.time() * 1000))

    trunks = {}
    trunks['words'] = post_data['words']
    trunks['appId'] = post_data['appId']
    trunks['timestamp'] = post_data['timestamp']
    post_data['sign'] = get_sign(trunks, appSecret_str)

    headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                "Content-Type": "application/json"
            }   
    
    #post_data_urlencode = urllib.urlencode(post_data)  #application/x-www-form-urlencoded 时才使用
    
    #print post_data

    request = urllib2.Request(url, json.dumps(post_data, ensure_ascii=False), headers) #content-type是json时，使用json.dumps；否则使用urllib.urlencode
    response = urllib2.urlopen(request)
    content = response.read()
    if (content):
        rs = json.loads(content)
        #print rs
    

if __name__ == '__main__':
    print datetime.datetime.now()
    #http_get_request()
    http_post_request()


    print datetime.datetime.now()
    #segment_test('我要请假')
