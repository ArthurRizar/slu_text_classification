#coding:utf-8
###################################################
# File Name: bert_data_utils.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月06日 星期三 14时04分26秒
#=============================================================
import sys
import codecs


sys.path.append('../')

from preprocess import tokenization

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def read_code_file(code_file):
    #get real label
    label2code = {}
    with codecs.open(code_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            line_info = line.split('\t')
            label = line_info[0].lower()
            code_value = line_info[1].lower()
            label2code[label] = code_value
    return label2code


def read_bert_labels_file(label_file):
    label_list = []
    with codecs.open(label_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            label_list.append(line)

    label_list = list(set(label_list))
    label2idx = {}
    idx2label = {}
    for (i, label) in enumerate(label_list):
        label2idx[label] = i
        idx2label[i] = label
        print(i, label)
    return label2idx, idx2label


def read_label_map_file(label_map_file):
    label_map = {}
    idx2label = {}
    with codecs.open(label_map_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            line_info = line.split('\t')
            idx = int(line_info[0])
            label = line_info[1]
            label_map[label] = idx
            idx2label[idx] = label
    return label_map, idx2label 



def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, is_predict=True):
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        pass
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    #print(example.label)

    label_id = None
    if example.label is not None:
        label_id = label_map[example.label]


    if ex_index < 5 and not is_predict:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label: %s (id = %d)" % (example.label.encode('utf8'), label_id))



    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
    return feature


def get_data_from_file(file_name):
    text_trunk = []
    with codecs.open(file_name, 'r', 'utf8') as fr:
        for i, line in enumerate(fr):
            line = line.strip().lower()
            line_info = line.split('\t')
            text = line_info[0].strip()
            label = line_info[1].strip()
            yield InputExample(guid=i, text_a=text, label=label)



def file_based_convert_examples_to_features(file_name, label_map, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    examples = get_data_from_file(file_name)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d" % (ex_index))

        feature = convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return examples, features
