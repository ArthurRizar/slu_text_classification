# slu_text_classification

A tf classic style (session run and feed dictionary) bert training and evaluating



# data
labes.tsv:
label1
label2
lable3
...

train.tsv:
sentence1\tlabel1
sentence2\tlabel2
....

test.tsv
(same format as train.tsv)

dev.tsv:
(same format as train.tsv)


# train
cd example
bash train_bert.sh

