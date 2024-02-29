import sys

import torch

sys.path.append('..')

from utils.nlgeval import calc_nlg_metrics

from DataReader import Dataset
import os
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers import HfArgumentParser
from transformers import BartTokenizer
from tqdm import tqdm
# load bart-based model
from transformers import BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartConfig
from transformers import BartTokenizer
from transformers.optimization import AdamW, Adafactor
import argparse
import warnings
import argparse
import copy
import json
import logging
import os
import pickle
import random
import time
from collections import defaultdict
import numpy as np
import torch
import transformers
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, BertTokenizer,BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser,DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,AutoModelForSeq2SeqLM, set_seed)
from transformers.trainer_utils import is_main_process
import sys
from datetime import datetime
import nltk
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", default="lcsts",type=str)
parser.add_argument("--lr2",default=1e-4,type=float)
parser.add_argument("--do_train",default=True)
parser.add_argument("--do_eval",default=True)
parser.add_argument("--do_predict",default=True)
parser.add_argument("--per_device_train_batch_size", default=16, type=int)
parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--warmup_ratio", default=0.0, type=float)
parser.add_argument("--max_source_length", default=512, type=int)
parser.add_argument("--generation_max_length", default=64, type=int)  # 这里可以改
parser.add_argument("--seed", default=3407, type=int)
parser.add_argument("--save_total_limit", default=3, type=int)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--metric_for_best_model", default="Bleu_2",type=str)
parser.add_argument("--greater_is_better", default=True)
parser.add_argument("--evaluation_strategy", default="epoch",type=str)  # 注意一下这个地方
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--save_strategy", default="epoch", type=str)
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--ignore_pad_token_for_loss", default=True)
parser.add_argument("--predict_with_generate", default=True)
parser.add_argument("--num_beams", default=4, type=int)
parser.add_argument("--not_pretrain", action="store_true")
parser.add_argument("--repetition_penalty", default=1.0, type=float)
# parser.add_argument("--config_path", default='../../MODEL/transformer_config', type=str)
parser.add_argument("--dataset_type", type=str, default="esconv", help="either be 'esconv' or 'P4G'")
parser.add_argument("--num_candidates", type=int, default=3, help="number of candidates")
parser.add_argument("--model_path", type=str, default="facebook/bart-base", help="model path")
parser.add_argument("--output_dir", type=str, default="./model/esconv/", help="output directory")

args = parser.parse_args()
arg_dict = args.__dict__

print(arg_dict)
logger = logging.getLogger(__name__)

args = parser.parse_args()
train_parser = HfArgumentParser(Seq2SeqTrainingArguments)

data_dir = '../TopicRanking/data/esconv/'
dataset_type = 'esconv'
num_candidates = 3
model_path = args.model_path
max_eval_samples = 100
output_dir = "./model/esconv/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


model, loading_info = BartForConditionalGeneration.from_pretrained(model_path, output_loading_info=True)
tokenizer = BartTokenizer.from_pretrained(model_path)

train_file = data_dir + 'train.json'
dev_file = data_dir + 'dev.json'
test_file = data_dir + 'test.json'

train_set = Dataset(tokenizer, train_file, num_candidates)
dev_set = Dataset(tokenizer, dev_file, num_candidates)
test_set = Dataset(tokenizer, test_file, num_candidates)
dev_set.total_data = dev_set.total_data[:max_eval_samples]

def compute_metrics(eval_preds):
    decoder_preds, decoder_labels = eval_preds
    ref_list = []
    hyp_list = []
    for ref, hyp in tqdm(zip(decoder_labels, decoder_preds[0])):
        preds_token_index = []
        for logits in hyp:
            token_index = logits.argmax(dim=-1).item()
            preds_token_index.append(token_index)
        ref = tokenizer.decode(ref, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        hyp = tokenizer.decode(preds_token_index, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if len(hyp) == 0:
            hyp = '&'
        ref_list.append(ref)
        hyp_list.append(hyp)
    # print 10 examples
    for i in range(10):
        print("ref:", ref_list[i])
        print("hyp:", hyp_list[i])
        print()
    return calc_nlg_metrics(decoder_preds=hyp_list, decoder_labels=ref_list)

def set_log(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


if __name__== "__main__":
    # define the optimizer (AdamW)
    optimizer = AdamW(model.parameters(), lr=args.lr2)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        overwrite_output_dir=args.overwrite_output_dir,
        warmup_ratio=args.warmup_ratio,
        generation_max_length=args.generation_max_length,
        seed=args.seed,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        predict_with_generate=args.predict_with_generate,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=test_set,
        eval_dataset=dev_set,
        optimizers=(optimizer, None)
    )
    # trainer.train()
    trainer.evaluate()
    # trainer.save_model("bart_topic_ranking")