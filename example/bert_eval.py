#coding:utf-8
###################################################
# File Name: eval.py
# Author: Meng Zhao
# mail: @
# Created Time: Fri 23 Mar 2018 09:27:09 AM CST
#=============================================================
import os
import sys
import csv
import codecs
import numpy as np

sys.path.append('../')


from preprocess import tokenization
from preprocess import bert_data_utils
from preprocess import dataloader
from config import *
from tensorflow.contrib import learn


#os.environ["CUDA_VISIBLE_DEVICES"] = "" # not use GPU


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("do_lower_case", True, "Whether to lower case the input text")
flags.DEFINE_string("vocab_file", BASE_DIR+"/example/runs/v"+VERSION+'/vocab.txt', "vocab file")
flags.DEFINE_string("label_file", BASE_DIR+"/example/runs/v"+VERSION+'/labels.txt', "label file")
flags.DEFINE_string("label_map_file", BASE_DIR+"/example/runs/v"+VERSION+'/label_map', "label map file")
#flags.DEFINE_string("init_checkpoint", BASE_DIR+"/example/runs/v"+VERSION+'/checkpoints/model.ckpt-3834', "vocab file")
flags.DEFINE_string("model_dir", BASE_DIR+"/example/runs/v"+VERSION+'/checkpoints', "vocab file")
flags.DEFINE_string("bert_config_file", BASE_DIR+"/example/runs/v"+VERSION+'/bert_config.json', "config json file")
tf.flags.DEFINE_string("test_data_file", '../data/test.tsv', "Test data source.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_sequence_length", 64, "max sequnce length")


tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def show_best_acc(preds, y_truth):
    correct_predictions = float(sum(preds==y_truth))
    print('Total number of test examples: {}'.format(len(y_truth)))
    print('Accuracy: {:g}'.format(correct_predictions/float(len(y_truth))))


def show_topk_acc(all_topk, y_truth):
    topk_corrects = 0.0
    errs_id_x = []
    errs_y = []
    for idx, y in enumerate(y_truth):
        if y in all_topk[idx]:
            topk_corrects += 1.0
        else:
            pred_y = all_topk[idx][0]
            errs_id_x.append(idx)
            errs_y.append((y, pred_y))
    print('Top k accuracy: {:g}'.format(topk_corrects/float(len(y_truth))))

def show_each_label_acc(idx2label, all_predictions, y_truth):
    accs_map = {}   
    statistic_map = {}
    for label_idx in idx2label:
        label = idx2label[label_idx]
        search_indices = [idx for idx, value in enumerate(y_truth) if value==label_idx]
        cur_y = y_truth[search_indices]
        cur_pred = all_predictions[search_indices]
        cur_correct_pred = float(sum(cur_pred==cur_y))
        if len(cur_y) != 0:
            accs_map[label_idx] = cur_correct_pred / float(len(cur_y))
            statistic_map[label_idx] = (int(cur_correct_pred), len(cur_y))
        else:
            accs_map[label_idx] = 0.0
            statistic_map[label_idx] = (0, 0)

    for label_idx in accs_map:
        cur_acc = accs_map[label_idx]
        corrects, total = statistic_map[label_idx]
        print('label_idx: {}, label: {}, corrects: {}, total: {}, Accuracy: {:g} '.format(label_idx, idx2label[label_idx], corrects, total, cur_acc))

def write_predictions(raw_examples, features, all_predictions, all_topk, idx2label):
    #get real label
    y_truth = np.array([item.label_id for item in features])
        
    #best acc
    show_best_acc(all_predictions, y_truth)
       

    #top k acc
    show_topk_acc(all_topk, y_truth)


    #each label acc
    show_each_label_acc(idx2label, all_predictions, y_truth)

    #save the evaluation to a csv
    all_topk_pred_label = []
    for indices in all_topk:
        labels = [idx2label[int(idx)] for idx in indices]
        all_topk_pred_label.append(labels)

    all_pred_label = [idx2label[int(idx)] for idx in all_predictions]
    utf8_x_raw = [item.text_a for item in raw_examples]
    utf8_y_raw = [idx2label[int(idx)] for idx in y_truth]
    #predictions_human_readable = np.column_stack((np.array(utf8_x_raw), all_pred_label))
    predictions_human_readable = np.column_stack((np.array(utf8_x_raw), utf8_y_raw))
    predictions_human_readable = np.column_stack((predictions_human_readable, all_topk_pred_label))

    out_path = os.path.join(model_dir, '..', 'prediciton.csv')
    print('Saving evaluation to {0}'.format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f, delimiter="\t",).writerows(predictions_human_readable)


def get_feed_data(features):
    feed_input_ids = [item.input_ids for item in features]
    feed_input_mask = [item.input_mask for item in features]
    feed_segment_ids = [item.segment_ids for item in features]

    return feed_input_ids, feed_input_mask, feed_segment_ids


def eval():
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    label_map, idx2label = bert_data_utils.read_label_map_file(FLAGS.label_map_file)
    raw_examples, features = bert_data_utils.file_based_convert_examples_to_features(FLAGS.test_data_file, 
                                                                                label_map,
                                                                                FLAGS.max_sequence_length,
                                                                                tokenizer)

    print('\nEvaluating...\n')

    #Evaluation
    checkpoint_file = tf.train.latest_checkpoint(model_dir)
    graph = tf.Graph()
    with graph.as_default():
        restore_graph_def = tf.GraphDef()
        restore_graph_def.ParseFromString(open(model_dir+'/frozen_model.pb', 'rb').read())
        tf.import_graph_def(restore_graph_def, name='')

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        
        with sess.as_default():
            #tensors we feed
            input_ids = graph.get_operation_by_name('input_ids').outputs[0]
            input_mask = graph.get_operation_by_name('input_mask').outputs[0]
            token_type_ids = graph.get_operation_by_name('segment_ids').outputs[0]
            is_training = graph.get_operation_by_name('is_training').outputs[0]
            
            #tensors we want to evaluate
            probs =  graph.get_operation_by_name('loss/probs').outputs[0]
            scores = graph.get_operation_by_name('loss/logits').outputs[0]
            pred_labels = graph.get_operation_by_name('loss/pred_labels').outputs[0]

            batches = dataloader.batch_iter(list(features), FLAGS.batch_size, 1, shuffle=False)

            #collect the predictions here
            all_predictions = []
            all_topk = []
            for batch in batches:
                feed_input_ids, feed_input_mask, feed_segment_ids = get_feed_data(batch)

                feed_dict = {input_ids: feed_input_ids,
                             input_mask: feed_input_mask,
                             token_type_ids: feed_segment_ids,
                             is_training: False,}

                batch_probs, batch_scores, batch_pred_labels = sess.run([probs, scores, pred_labels],
                                                                        feed_dict)
                batch_pred_label = np.argmax(batch_probs, -1)
                all_predictions = np.concatenate([all_predictions, batch_pred_label])
                temp = np.argsort(-batch_scores, 1)
                all_topk.extend(temp[:, :3].tolist()) #top 3
                

    raw_examples = list(bert_data_utils.get_data_from_file(FLAGS.test_data_file))
    truth_label_ids = np.array([item.label_id for item in features])
    #write predictions to file
    write_predictions(raw_examples, features, all_predictions, all_topk, idx2label)


if __name__ == '__main__':
    eval()
