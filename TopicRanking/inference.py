import argparse
import os
import json

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import sys
sys.path.append('../../')
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers import HfArgumentParser
import copy
# from torch.utils.data.dataset import Dataset
from DataReader import get_stratege, read_pk, TopicRankingDataset
from PairedBert import MODEL_LIST
from transformers import BertTokenizer
from train import SeqTrain, compute_metrics_with_ranking_result, ranking_label_map
import warnings
from collections import defaultdict
from transformers.trainer import is_datasets_available,IterableDatasetShard

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='output/esconv/',
                        help='Path to load the model')
parser.add_argument('--inference_data_path', default='data/esconv/test.json',
                    help='Path to load the data')
parser.add_argument('--output_path', default='inference_results/esconv/test.json',
                        help='Path to save the output')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--model_type", default="1")
parser.add_argument("--cluster_num", default=6)

args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = MODEL_LIST[args.model_type].from_pretrained(args.model_path)

if 'P4G' in args.inference_data_path:
    test_set = TopicRankingDataset(args.inference_data_path, tokenizer, data_type='p4g')
else:
    test_set = TopicRankingDataset(args.inference_data_path, tokenizer, cluster_num=args.cluster_num)

# parse args.output_path and create the directory if it does not exist
args.output_dir = os.path.dirname(args.output_path)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def inference_on_data():
    for sample in test_set:
        print(sample.keys())
        break
    training_args = TrainingArguments(
        per_device_eval_batch_size=args.batch_size,
        output_dir=args.output_dir,
        do_train=False,
        do_eval=False,
        do_predict=True,
    )
    trainer = SeqTrain(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics_with_ranking_result,
        eval_dataset=test_set,
    )
    outputs = trainer.predict(test_set)
    labels = outputs.label_ids
    preds = outputs.predictions
    metrics = outputs.metrics
    # print the metrics
    print()
    print("#"*50)
    print("Metrics on the test set: ", metrics)
    # output the results to a file
    query_dict = defaultdict(list)
    for pre_prob, ql in zip(preds, labels):
        label, qid = ql
        query_dict[qid].append((pre_prob, label))
    ranking_results = []
    for q, pl in query_dict.items():
        ranking_len = len(pl)
        sorted_prob_label = sorted(pl, key=lambda x: x[0], reverse=True)
        pred_labels = [int(ranking_label_map(x[1], ranking_len)) for x in sorted_prob_label]
        ranking_results.append(pred_labels)

    with open(args.inference_data_path, 'r') as f:
        data = json.load(f)

    valid_data = []
    for sample_index, sample in enumerate(data):
        try:
            for one_topic in sample['topic_candidates']:
                topic_sen = one_topic['topic']
                tmp_label = one_topic['ranking_scores']['ranking']
            valid_data.append(sample)
        except:
            continue

    for sample_index, sample in enumerate(valid_data):
        sample['ranking_result'] = ranking_results[sample_index]
        del sample['progression_info']

    with open(args.output_path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    inference_on_data()