#coding:utf-8
###################################################
# File Name: tf_serving_start.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年04月30日 星期二 13时33分47秒
#=============================================================
source activate tensorflow_new_3.6

source $PWD/../global_config.cfg

export REST_API_PORT=$rest_api_port
export MODEL_DIR=$PWD/../runs/v1.0/checkpoints/
export MODEL_NAME=default


nohup tensorflow_model_server --rest_api_port=$REST_API_PORT \
                              --model_name=$MODEL_NAME \
                              --model_base_path=$MODEL_DIR \
      >output.file 2>&1 &

nohup python tf_serving_client.py> output_client.file 2>&1 &
