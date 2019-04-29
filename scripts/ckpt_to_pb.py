#coding:utf-8
###################################################
# File Name: test.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月11日 星期一 15时16分58秒
#=============================================================
import os
import sys
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants


sys.path.append('../')

from setting import *




def describe_graph(graph_def, show_nodes=False):
  print('Input Feature Nodes: {}'.format(
      [node.name for node in graph_def.node if node.op=='Placeholder']))
  print('')
  print('Unused Nodes: {}'.format(
      [node.name for node in graph_def.node if 'unused'  in node.name]))
  print('')
  print('Output Nodes: {}'.format( 
      [node.name for node in graph_def.node if (
          'predictions' in node.name or 'softmax' in node.name)]))
  print('')
  print('Quantization Nodes: {}'.format(
      [node.name for node in graph_def.node if 'quant' in node.name]))
  print('')
  print('Constant Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Const'])))
  print('')
  print('Variable Count: {}'.format(
      len([node for node in graph_def.node if 'Variable' in node.op])))
  print('')
  print('Identity Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Identity'])))
  print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

  if show_nodes==True:
    for node in graph_def.node:
      print('Op:{} - Name: {}'.format(node.op, node.name))

def get_graph_def_from_saved_model(saved_model_dir): 
  with tf.Session() as session:
    meta_graph_def = tf.saved_model.loader.load(
    session,
    tags=[tag_constants.SERVING],
    export_dir=saved_model_dir
  ) 
  return meta_graph_def.graph_def


def get_size(model_dir, model_file='saved_model.pb'):
  model_file_path = os.path.join(model_dir, model_file)
  print(model_file_path, '')
  pb_size = os.path.getsize(model_file_path)
  variables_size = 0
  if os.path.exists(
      os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
    variables_size = os.path.getsize(os.path.join(
        model_dir,'variables/variables.data-00000-of-00001'))
    variables_size += os.path.getsize(os.path.join(
        model_dir,'variables/variables.index'))
  print('Model size: {} KB'.format(round(pb_size/(1024.0),3)))
  print('Variables size: {} KB'.format(round( variables_size/(1024.0),3)))
  print('Total Size: {} KB'.format(round((pb_size + variables_size)/(1024.0),3)))

def freeze_graph(input_checkpoint_dir, output_graph):
    '''
    :param input_checkpoint_dir:
    :param output_graph: PB模型保存路径
    :return:
    '''
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
    output_node_names = ['input_ids', 'input_mask', 'segment_ids', 'is_training', 'loss/pred_labels', 'loss/probs', 'loss/logits']
    cp_file = tf.train.latest_checkpoint(input_checkpoint_dir)
    saver = tf.train.import_meta_graph('{}.meta'.format(cp_file))


    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, cp_file) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names)# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

if __name__ == '__main__':
  #input_checkpoint = MODEL_DIR + '/checkpoints/model-259'
  input_checkpoint = MODEL_DIR + '/checkpoints/'
  out_pb_path = MODEL_DIR + '/checkpoints/frozen_model.pb'
  freeze_graph(input_checkpoint, out_pb_path)


  #test = get_graph_def_from_saved_model('../example/runs/v0.91/checkpoints')
  test = get_size('../example/runs/v0.91/checkpoints', 'frozen_model.pb')


