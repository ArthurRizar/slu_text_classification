###################################################################
# File Name: start.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月12日 星期二 16时19分57秒
#=============================================================
#!/bin/bash
source activate tensorflow_new_3.6
python bert_train.py --task_name=test \
                     --output_dir=../runs/v1.0 \
                     --data_dir=../data \
                     --init_checkpoint=/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12/bert_model.ckpt \
                     --bert_config_file=/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12/bert_config.json \
                     --vocab_file=/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12/vocab.txt \
                     --max_seq_length=64  \
                     --do_train=true \
                     --num_train_epochs=1; 

python ckpt_to_pb.py
