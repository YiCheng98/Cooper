import argparse
import logging
import math
import os
import pickle
import random
import json

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../../')
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support,mean_absolute_error
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
# from clss_trainer import MyTrainer as TrainerHfArgumentParserCounter
import warnings
warnings.filterwarnings("ignore")
'''
bert feedback predict
'''
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_model',default='bert-base-uncased',
                        help='Pretrain model weight')
parser.add_argument('--output_dir', default='output/esconv',
                        help='The output directory where the model predictions and checkpoints will be written.')
parser.add_argument('--data_dir', default='data/esconv',
                        help='Path saved data')
parser.add_argument('--seed',default=42,
                        help='Path saved data')
parser.add_argument('--per_device_train_batch_size', default=16, type=int)
parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
# parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
parser.add_argument('--num_train_epochs', default=5, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--lr2', default=5e-5, type=float)
parser.add_argument('--evaluation_strategy', default="epoch", type=str)
parser.add_argument('--save_strategy', default="epoch", type=str)
parser.add_argument('--do_train', default=True)
parser.add_argument('--do_eval', default=True)
parser.add_argument('--do_predict', default=True)
parser.add_argument('--load_best_model_at_end', default=True)
parser.add_argument("--metric_for_best_model", default="topacc@3")
parser.add_argument("--save_total_limit", default=3, type=int)
parser.add_argument("--model_type", default="1")
parser.add_argument("--cluster_num", default=6)
parser.add_argument("--optim_type", default="2")



# parser.add_argument('--load_best_model_at_end', default=True)
args = parser.parse_args()
# print(args.extend_data, args.output_dir)
# args.output_dir = f'rebase_03/{args.output_dir}_model{args.model_type}_cluster_{args.cluster_num}_optim_{args.optim_type}'
# args.output_dir = f'seventh_try/{args.output_dir}_model{args.model_type}_cluster_{args.cluster_num}'
args.logging_dir = args.output_dir + "/runs"

# strateges = get_stratege('../new_strategy.json', norm=True)
# stratege2id = {v: k for k, v in enumerate(strateges)}
train_path = args.data_dir + '/train.json'
val_path = args.data_dir + '/dev.json'
test_path = args.data_dir + '/test.json'
tokenizer = BertTokenizer.from_pretrained(args.pretrain_model, use_fast=False)

model, loading_info = MODEL_LIST[args.model_type].from_pretrained(args.pretrain_model, num_labels=1, problem_type="regression",
                                                          output_loading_info=True)
sencond_parameters = loading_info['missing_keys']

if 'P4G' in args.data_dir:
    train_set = TopicRankingDataset(train_path, tokenizer, data_type='p4g')
    eval_set = TopicRankingDataset(val_path, tokenizer, data_type='p4g')
    test_set = TopicRankingDataset(test_path, tokenizer, data_type='p4g')
else:
    train_set = TopicRankingDataset(train_path, tokenizer, cluster_num=args.cluster_num)
    eval_set = TopicRankingDataset(val_path, tokenizer, cluster_num=args.cluster_num)
    test_set = TopicRankingDataset(test_path, tokenizer, cluster_num=args.cluster_num)

print(f"output_dir: {args.output_dir}")
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def compute_metrics_with_raning_result_before(result):
    labels = result.label_ids
    preds = result.predictions
    query_dict = defaultdict(list)
    for pre_prob, ql in zip(preds, labels):
        label, qid = ql
        query_dict[qid].append((pre_prob, label))
    pos_num,neg_num = 0, 0
    for q in query_dict:
        v_list = query_dict[q]
        v_len = len(v_list)
        for i in range(v_len):
            for j in range(i+1, v_len):
                pre_flag = v_list[i][0] - v_list[j][0]
                label_flag = v_list[i][1] - v_list[j][1]
                if pre_flag * label_flag > 0:
                    pos_num += 1
                if pre_flag * label_flag < 0:
                    neg_num += 1
    # totalnum = pos_num + neg_num
    pnr = pos_num / neg_num if neg_num > 0 else 0
    return {'pnr':pnr}

def compute_metrics_with_ranking_result(result):
    # TOP1 ACC, MAP@3, MAP@5, DCG@3, DCG@5.
    labels = result.label_ids
    preds = result.predictions
    query_dict = defaultdict(list)
    for pre_prob, ql in zip(preds, labels):
        label, qid = ql
        query_dict[qid].append((pre_prob, label))
    top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = 0, 0, 0,0,0
    dcg3, edcg3 = 0, 0
    dcg5, edcg5 = 0, 0
    mrr_score = 0
    num = 0
    for q, pl in query_dict.items():
        num += 1
        sorted_prob_label = sorted(pl, key=lambda x: x[0], reverse=True)
        if num < 5:
            print(q, sorted_prob_label[:5])
        pred_labels = [x[1] for x in sorted_prob_label]

        for i, pi in enumerate(pred_labels):
            if pi == len(sorted_prob_label):
                mrr_score += 1/(i+1)
                break
        top1_acc += pred_labels[0] == len(sorted_prob_label)
        top2_acc += np.sum(np.array(pred_labels[:2]) == len(sorted_prob_label))
        top3_acc += np.sum(np.array(pred_labels[:3]) == len(sorted_prob_label))
        top4_acc += np.sum(np.array(pred_labels[:4]) == len(sorted_prob_label))
        top5_acc += np.sum(np.array(pred_labels[:5]) == len(sorted_prob_label))
        # idcg3 = np.sum(np.arange(len(sorted_prob_label)-3, len(sorted_prob_label))
        dcg3 += np.sum(np.array(pred_labels[:3])) / np.sum(np.arange(len(sorted_prob_label)-3, len(sorted_prob_label)) )
        edcg3 += np.sum(np.exp2(np.array(pred_labels[:3]))-3) / np.sum(np.exp2(np.arange(len(sorted_prob_label)-3, len(sorted_prob_label))) -3)

        dcg5 += np.sum(np.array(pred_labels[:5])) / np.sum(np.arange(len(sorted_prob_label)-5, len(sorted_prob_label)) )
        edcg5 += np.sum(np.exp2(np.array(pred_labels[:5]))-5) / np.sum(np.exp2(np.arange(len(sorted_prob_label)-5, len(sorted_prob_label))) -5)

# 需要定义清楚， gt是top3还是top1？ 如果是top1，那么为什么生成的时候使用top3？ 如果是top3， 那么这是怎么定义的？

    ans = {
        "topacc@1": top1_acc / len(query_dict),
        "topacc@2": top2_acc / len(query_dict),
        "topacc@3": top3_acc / len(query_dict),
        "topacc@4": top4_acc / len(query_dict),
        "topacc@5": top5_acc / len(query_dict),
        "dcg@3": dcg3 / len(query_dict),
        "edcg@3": edcg3 / len(query_dict),
        "dcg@5": dcg5 / len(query_dict),
        "edcg@5": edcg5 / len(query_dict),
        "mrr": mrr_score / len(query_dict)
    }
    return ans

def tmp_socre(result):
    return {"ab": 1.0}

def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

from transformers.optimization import AdamW, Adafactor

def get_optimer(model, second_parameter, train_parser):
    # decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "lr": args.lr2,
        }
    ]
    if args.optim_type == '0':# sgd
        optimizer_kwargs = {}
        optimizer_cls = torch.optim.SGD
    elif args.optim_type == '1': # RMSprop
        optimizer_kwargs = {}
        optimizer_cls = torch.optim.RMSprop
    elif args.optim_type == "2": # Adamw
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (train_parser.adam_beta1, train_parser.adam_beta2),
            "eps": train_parser.adam_epsilon,
        }
    # optimizer_kwargs["lr"] = train_parser.learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.trainer import is_datasets_available,IterableDatasetShard
# import datasets

from torch.utils.data import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

class QuerySequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    # data_source: Sized

    def __init__(self, data_source) -> None:
        self.data_source = data_source


    def __iter__(self) -> Iterator[int]:
        random.shuffle(self.data_source.index_list)
        for x in self.data_source.index_list:
            for i in range(x[0],x[1]):
                yield i
        # return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
class SeqTrain(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = QuerySequentialSampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def train():
    # remove the following arguments from args: ['cluster_num', 'data_dir', 'lr2', 'model_type', 'optim_type', 'pretrain_model']
    args_copy = copy.deepcopy(args)
    for k in ['cluster_num', 'data_dir', 'lr2', 'model_type', 'optim_type', 'pretrain_model']:
        del args_copy.__dict__[k]
    training_args = HfArgumentParser(TrainingArguments).parse_dict(vars(args_copy))[0]
    # add the following argument TrainingArguments: ['cluster_num', 'data_dir', 'lr2', 'model_type', 'optim_type', 'pretrain_model']
    for k in ['cluster_num', 'data_dir', 'lr2', 'model_type', 'optim_type', 'pretrain_model']:
        setattr(training_args, k, getattr(args, k))

    optimer = get_optimer(model, sencond_parameters,training_args)
    trainer = SeqTrain(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_with_ranking_result,
        train_dataset=train_set,
        eval_dataset=eval_set,
        optimizers=(optimer,None),
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()
    predict_metric = trainer.evaluate(test_set, metric_key_prefix="predict")

    print(predict_metric)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def test_one_case():
    for i in range(3):
        one_case  = train_set.__getitem__(i)
        output = model(**one_case)
        one_logtis = output[1]
        print(one_case)
        print('logits: ', one_logtis)
        print('#'* 50)


if __name__ == '__main__':
    os.environ["WANDB_DISABLED"] = "true"
    fix_random(args.seed)
    train()