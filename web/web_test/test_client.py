#coding:utf-8
###################################################
# File Name: test_client.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月11日 星期一 17时57分43秒
#=============================================================
import tensorflow as tf
sess = tf.Session("grpc://localhost:45795")

print(sess)
