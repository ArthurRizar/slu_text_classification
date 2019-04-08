#coding:utf-8
###################################################
# File Name: segment_client.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年06月11日 星期一 14时03分41秒
#=============================================================


#Python 2.7
import json
import hashlib
import time
import datetime
import urllib

class SegClient(object):
    def __init__(self, settings):
        self.app_id = settings['appId']
        self.app_secret = settings['appSecret']
        self.client_url = settings['url']
        self.headers = { 
                        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                        "Content-Type": "application/json"
                    }



    def gen_request_key(self, trunks):
        rs = ''
        return rs 


    def segment_text(self, text):
        '''
        brief: 返回json字符串
        '''
        post_data = {}
        
        post_data['appId'] = self.app_id
        post_data['words'] = text

        timestamp = str(int(time.time()) * 1000)
        post_data['timestamp'] = timestamp

        trunks = []
        trunks.append(('timestamp', timestamp))
        trunks.append(('words', text))
        post_data['sign'] = self.gen_request_key(trunks)
        
        #request web url
        request = urllib2.Request(self.client_url, json.dumps(post_data, ensure_ascii=False), self.headers)
        response = urllib2.urlopen(request)
        content = response.read()
        if (content):
            rs = json.loads(content)
            return rs
    
    def segment_text2segs(self, text):
        '''
        brief: return a utf-8 string which have all the segs
        '''
        
        segs_json = self.segment_text(text)
        segs = segs_json['data']

        segs = json.loads(segs)
        rs = ''
        for item in segs:
            word = item['word']
            POS = item['nature']
            rs += word + '/' + POS + ' '
        rs = rs.strip()
        return rs.encode('utf-8') 

if __name__ == '__main__':
    pass
