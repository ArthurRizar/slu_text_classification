export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=test \
  --do_train=true \
  --data_dir=data/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=output/
  
  
  
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
