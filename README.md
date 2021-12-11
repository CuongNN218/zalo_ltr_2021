# zalo_ltr_2021
Source code for Zalo AI 2021 submission

# Solution:
## Train BM 25
To train BM25: ``python bm25_train.py``
Use load_docs to save time for later run: ``python bm25_train.py --load_docs``

To evaluate: ``python bm25_create_pairs.py``
This step will also create top_k negative pairs from BM25. We choose top_k= 20, 50
Pairs will be saved to: pair_data/

These pairs will be used to train round 1 Sentence Transformer model

## Create corpus: 
Run ``python create_corpus.txt``
This step will create:
- corpus.txt  (for finetune language model)
- cocondenser_data.json (for finetune CoCondenser model)

## Finetune language model using Huggingface
Pretrained model:
- viBERT
- vELECTRA
- phobert-base
- phobert-large

$MODEL_NAME= phobert-large
$DATA_FILE= corpus.txt
$SAVE_DIR=

Run ``python run_mlm.py --model_name_or_path $MODEL_NAME --train_file $DATA_FILE --do_train --do_eval --output_dir $SAVE_DIR --line_by_line --overwrite_output_dir --save_steps 2000 --num_train_epochs 20 --per_device_eval_batch_size 32 --per_device_train_batch_size 32``

##  Train condenser and cocondenser from language model checkpoint
Original source code here: https://github.com/luyug/Condenser

### Create data for Condenser: 
Run: ``python helper/create_train.py --tokenizer_name $MODEL_NAME --file $DATA_FILE --save_to $SAVE_CONDENSER --max_len $MAX_LENGTH``
$MODEL_NAME=vinai/phobert-large
$MAX_LENGTH=256
$DATA_FILE=../generated_data/corpus.txt
$SAVE_CONDENSER=../generated_data/

$MODEL_NAME checkpoint from finetuned language model 

```
python run_pre_training.py \
  --output_dir $OUTDIR \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --save_steps 2000 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --fp16 \
  --warmup_ratio 0.1 \
  --learning_rate 5e-5 \
  --num_train_epochs 8 \
  --overwrite_output_dir \
  --dataloader_num_workers 32 \
  --n_head_layers 2 \
  --skip_from 6 \
  --max_seq_length $MAX_LENGTH \
  --train_dir $SAVE_CONDENSER \
  --weight_decay 0.01 \
  --late_mlm
```

``python run_pre_training.py   --output_dir saved_model_1/   --model_name_or_path ../Legal_Text_Retrieval/lm/large/checkpoint-30000   --do_train   --save_steps 2000   --per_device_train_batch_size 32   --gradient_accumulation_steps 4   --fp16   --warmup_ratio 0.1   --learning_rate 5e-5   --num_train_epochs 8   --overwrite_output_dir   --dataloader_num_workers 32   --n_head_layers 2   --skip_from 6   --max_seq_length 256   --train_dir ../generated_data/   --weight_decay 0.01   --late_mlm``


Train cocodenser:
python  run_co_pre_training.py   --output_dir saved_model/cocondenser/   --model_name_or_path $CODENSER_CKPT   --do_train   --save_steps 2000   --model_type bert   --per_device_train_batch_size 32   --gradient_accumulation_steps 1   --fp16   --warmup_ratio 0.1   --learning_rate 5e-5   --num_train_epochs 10   --dataloader_drop_last   --overwrite_output_dir   --dataloader_num_workers 32   --n_head_layers 2   --skip_from 6   --max_seq_length 256   --train_dir ../generated_data/cocondenser/   --weight_decay 0.01   --late_mlm  --cache_chunk_size 32 --save_total_limit 1

## Train Sentence Transformer
Round 1: using BM25 negative pairs
``python train_sentence_bert.py``
Round 2: using hard negative pairs create from Round 1 model
Run ``python hard_negative_mining.py``

Note: Use cls_pooling for condenser


