import argparse
import logging
import math
import os
import pickle
import random
import json

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import sys
sys.path.append('../../')
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support,mean_absolute_error
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
import sklearn.metrics
from transformers.trainer import Trainer
# from clss_trainer import MyTrainer as Trainer
from transformers.training_args import TrainingArguments
from transformers import HfArgumentParser
import copy
# from torch.utils.data.dataset import Dataset
from DataReader import get_stratege, read_pk, TopicRankingDataset
from PairedBert import MODEL_LIST
from transformers import BertTokenizer
import warnings
from collections import defaultdict, Counter

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='output/esconv/',
                        help='Path to load the model')
parser.add_argument('--inference_data_path', default='data/esconv/test.json',
                    help='Path to load the data')
parser.add_argument('--output_dir', default='inference_results/esconv/ranking_results.json',
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

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def inference_on_data():
    model.eval()
    all_results = []
    with torch.no_grad():
        for sample in test_set:
            sample = test_set.__getitem__(1)
            output = model(**sample)
            logits = output[1]
            # print(one_case)
            print('logits: ', logits)
            print('#' * 50)
    print('all_results: ', all_results)


if __name__ == '__main__':
    inference_on_data()