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
from DataReader import get_stratege, read_pk, TopicRankingDataset, TopicRankingDatasetForTest
# from PairedBert import BertForTopicRanking, BertForTopicRankingWoProgression
from transformers import BertTokenizer
import warnings
from collections import defaultdict, Counter
warnings.filterwarnings("ignore")
'''
bert feedback predict
'''
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_model', default='../MODEL/bert-base-uncased',
                        help='Pretrain model weight')
parser.add_argument('--output_dir', default='./test_each_metric/',
                        help='The output directory where the model predictions and checkpoints will be written.')
parser.add_argument('--data_dir', default='./data/',
                        help='Path saved data')
parser.add_argument('--seed', default=42,
                        help='Path saved data')
parser.add_argument('--per_device_train_batch_size', default=16, type=int)
parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
# parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
parser.add_argument('--num_train_epochs', default=12, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--lr2', default=5e-5, type=float)
parser.add_argument('--evaluation_strategy', default="epoch", type=str)
parser.add_argument('--save_strategy', default="epoch", type=str)
parser.add_argument('--do_train', default=True)
parser.add_argument('--do_eval', default=True)
parser.add_argument('--do_predict', default=True)
parser.add_argument('--load_best_model_at_end', default=True)
parser.add_argument("--metric_for_best_model", default="map@3")
parser.add_argument("--save_total_limit", default=2, type=int)
parser.add_argument("--add_progression", default="true")


# parser.add_argument('--load_best_model_at_end', default=True)
args = parser.parse_args()
# print(args.extend_data, args.output_dir)
args.output_dir = f'{args.output_dir}/rank_output'

# strateges = get_stratege('../new_strategy.json', norm=True)
# stratege2id = {v: k for k, v in enumerate(strateges)}
# train_path = args.data_dir + 'train.json'
# val_path = args.data_dir + 'dev.json'
# test_path = args.data_dir + 'test.json'

# tokenizer.add_tokens(list(stratege2id.keys()))

# Bertmodel = BERTMODEL_LIST[args.model_type]
# BertDataset = PredictFeedBackDataset
# if args.add_progression == "true":
#     print('add progression is True')
#     model, loading_info = BertForTopicRanking.from_pretrained(args.pretrain_model, num_labels=1, problem_type="regression",
#                                                     output_loading_info=True)
# else:
#     print('add progression is False')
#     model, loading_info = BertForTopicRankingWoProgression.from_pretrained(args.pretrain_model, num_labels=1, problem_type="regression",
#                                                     output_loading_info=True)




# train_set = TopicRankingDataset(train_path, tokenizer)
# eval_set = TopicRankingDataset(val_path, tokenizer)
# test_set = TopicRankingDataset(test_path, tokenizer)
# print(args.output_dir)

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
    top1_right = 0
    top3_ap, top5_ap = 0,0
    dcg3, dcg5 = 0, 0
    num = 0
    for q, pl in query_dict.items():
        num += 1
        sorted_prob_label = sorted(pl, key=lambda x: x[0], reverse=True)
        if num < 50:
            print(q, sorted_prob_label[:5])
        pred_labels = [x[1] for x in sorted_prob_label]
        top1_right += pred_labels[0] == len(sorted_prob_label)
        top3_right = np.sum(np.array(pred_labels[:3]) >= len(sorted_prob_label) - 3)
        top5_right = np.sum(np.array(pred_labels[:5]) >= len(sorted_prob_label) - 5)
        top3_ap += top3_right / 3
        top5_ap += top5_right / 5
        dcg3 += np.sum(np.array(pred_labels[:3]) / np.log2(np.arange(3)+2))
        dcg5 += np.sum(np.array(pred_labels[:5]) / np.log2(np.arange(5)+2))

    ans = {
        "top1_acc": top1_right / len(query_dict),
        "map@3": top3_ap / len(query_dict),
        "map@5": top5_ap / len(query_dict),
        "dcg@3": dcg3 / len(query_dict),
        "dcg@5": dcg5 / len(query_dict)
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
            "params": [p for n, p in model.named_parameters() if n in second_parameter],
            "lr": args.lr2,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in second_parameter],
            "lr": args.learning_rate
        },
    ]
    optimizer_cls = Adafactor if train_parser.adafactor else AdamW
    if train_parser.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
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

        # train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

#
# def train():
#
#
#     training_args = HfArgumentParser(TrainingArguments).parse_dict(vars(args))[0]
#     optimer = get_optimer(model, sencond_parameters,training_args)
#     trainer = SeqTrain(
#         model=model,
#         args=training_args,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics_with_ranking_result,
#         train_dataset=train_set,
#         eval_dataset=eval_set,
#         optimizers=(optimer,None),
#     )
#
#     # Training
#     train_result = trainer.train()
#     trainer.save_model()
#     predict_metric = trainer.evaluate(test_set, metric_key_prefix="predict")
#
#     print(predict_metric)
#     metrics = train_result.metrics
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)
#     trainer.save_state()

# dialogue_history, topic_list, current_state, topk_cluster_index, tokenizer

def make_tmp_dataset(dialogue_history, topic_list, current_state, distance_list):
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model, use_fast=False)

    input_dict = {
        "dialogue_history": dialogue_history,
        "topic_list": topic_list,
        "current_state": list(current_state),
        "distance_list": distance_list,
        "tokenizer":tokenizer
    }

    passed_dataset1 = TopicRankingDatasetForTest(**input_dict)
    return passed_dataset1

#dialogue_history, topic_list, current_state, topk_cluster_index
'''
{'dialogue_history': 'usr: Hello good afternoon .', 
'topic_list': ['How have you been feeling lately? Is there anything specific that has been bothering you or causing you distress?"', 'Is there anything you would like to share about what you\'ve been going through or any challenges you\'ve been facing?"', 'Can you tell me more about any recent events or situations that have been difficult for you?"', 'Passively responding to the seeker according to their last utterance.'], 
'current_state': [[],[],[]] 其中3个list，代表三个agent的state。
‘distance_list’:  # 代表各个维度的state到各个聚类中心的距离。
current_state与distance_list与数据集中的形式一致。
load_model_path: 加载模型位置。
model_type: 模型种类， 默认‘20’
'''
def test_one_case(dialogue_history, topic_list, current_state, distance_list,load_model_path, model_type="20"):
    def tmp_metric(result):
        return {"acc":0}
    from PairedBert import MODEL_LIST
    tokenizer = BertTokenizer.from_pretrained(load_model_path, use_fast=False)
    tmp_dataset = make_tmp_dataset(dialogue_history, topic_list, current_state, distance_list)
    training_args = HfArgumentParser(TrainingArguments).parse_dict(vars(args))[0]
    model, loading_info = MODEL_LIST[model_type].from_pretrained(load_model_path, num_labels=1, problem_type="regression",
                                                    output_loading_info=True)
    sencond_parameters = loading_info['missing_keys']
    optimer = get_optimer(model, sencond_parameters,training_args)
    trainer1 = SeqTrain(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=tmp_metric,
        train_dataset=tmp_dataset,
        eval_dataset=tmp_dataset,
        optimizers=(optimer,None),
    )
    predict = trainer1.predict(tmp_dataset)
    print(predict[0], len(predict[0]))
    topic_and_score = dict(zip(topic_list, predict[0]))
    sorted_topic = sorted(topic_and_score.items(), key=lambda x:x[1], reverse=True)
    return sorted_topic


    # for i in range(3):
    #     one_case  = train_set.__getitem__(i)
    #     output = model(**one_case)
    #     one_logtis = output[1]
    #     print(one_case)
    #     print('logits: ', one_logtis)
    #     print('#'* 50)


if __name__ == '__main__':
    os.environ["WANDB_DISABLED"] = "true"
    fix_random(args.seed)
    # train()
    test_one_case()