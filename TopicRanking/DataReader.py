import copy
import json
import random
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset,IterableDataset
from transformers import BartTokenizer
from collections import defaultdict,Counter
from sklearn.metrics import accuracy_score
def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# def load_json(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             data.append(json.loads(line.strip()))
#     return data
def load_json(file_path):
    with open(file_path, 'r') as f:
        tmp_data = json.load(f)
    return tmp_data

def read_json(file_path):
    with open(file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_pk(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def norm_strategy(strategy):
    norm_str = "-".join(strategy.split())
    return "@["+norm_str+"]"


def get_stratege(file_path, norm=False):
    with open(file_path,'r', encoding='utf-8') as f:
        data = json.load(f)
    data = [d.replace('[','').replace(']','') for d in data]
    if norm:
        data = [norm_strategy(d) for d in data]
    print('strategy: ', data)

    return data


def _norm(x):
    return ' '.join(x.strip().split()).lower()



class TopicRankingDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_history_len=128, max_topic_len=16, data_type='esc',cluster_num=15):
        super(TopicRankingDataset, self).__init__()
        self.max_history_len = max_history_len
        self.max_topic_len = max_topic_len
        self.cluster_num = int(cluster_num)
        self.tokenizer = tokenizer
        data = load_json(file_path)
        self.total_data = []
        # item_id
        topic_candidate_num_distribution = []
        label_distribution = []
        self.index_list = []
        start_index = 0
        continue_num = 0
        for item_id, one_item in enumerate(data):
            #  one_item['dialogue_history']
            current_state = []
            topk_cluster_index = []
            if data_type == 'esc':
                state_name = ['exploration_state','comfort_state','action_state']
                sub_task_name = {"Exploration":0,"Comfort":1,"Action":2,"Passive Response":1}
            else:
                state_name = ['inquiry_state','appeal_state','proposition_state']
                sub_task_name = {"inquiry":0,"appeal":1,"proposition":2,"passive_response":1}
                self.cluster_num = 6

            for one_name in state_name:
                current_state.append(one_item['progression_info'][one_name]['state_embedding'][0])
                tmp_distance_list = one_item['progression_info'][one_name]['distances']
                topk_cluster_index.append(np.argsort(tmp_distance_list)[-self.cluster_num:])
            topic_candidate_num_distribution.append(len(one_item['topic_candidates']))
            # print('whats your problem', len(one_item['topic_candidates']))
            try:
                for one_topic in one_item['topic_candidates']:
                    topic_sen = one_topic['topic']
                    # sub_task = one_topic['subgoal']
                    # print('keys: ', one_topic.keys())

                    tmp_label = len(one_item['topic_candidates']) - one_topic['ranking_scores']['ranking'] + 1
                    label_distribution.append(tmp_label)
                    self.total_data.append({
                        "item_id":item_id,
                        "history": one_item['dialogue_history'],
                        "topic":topic_sen,
                        "sub_task":sub_task_name[one_topic['subgoal']],
                        "labels": tmp_label,
                        "current_state":current_state,
                        "topk_cluster_index":topk_cluster_index
                    })
            except:
                continue_num += 1
                continue
            self.index_list.append((start_index, start_index + len(one_item['topic_candidates'])))
            start_index = start_index + len(one_item['topic_candidates'])
        print('continue_Num: ', continue_num)
        print('topic_candidate: ', Counter(topic_candidate_num_distribution).most_common(10))
        print('label distribution: ', Counter(label_distribution).most_common(20))
        print('total trained num: ', len(self.total_data))
        # if 'test' in file_path:
        #     self.total_data = self.total_data[:1000]
    def encode_sentence(self, history, topic):
        history_token = self.tokenizer.encode(history, add_special_tokens=False)
        topic_token = self.tokenizer.encode(topic, add_special_tokens=False)
        input_ids = [self.tokenizer.cls_token_id] + history_token[-self.max_history_len:] + \
                    [self.tokenizer.sep_token_id] + topic_token[:self.max_topic_len] + [self.tokenizer.sep_token_id]
        return self.padding_sentence(input_ids)

    def padding_sentence(self, input_list):
        max_len = self.max_topic_len + self.max_history_len + 3
        return input_list + [self.tokenizer.pad_token_id] * max(max_len - len(input_list), 0)

    def get_str_form(self, tmp_str, tmp_len):
        str_list = tmp_str.split()[:tmp_len]
        return ' '.join(str_list)


    def get_model_input(self, tmp_dic):
        ans = {
            "input_ids": torch.tensor(self.encode_sentence(tmp_dic['history'], tmp_dic['topic'])),
            # "sub_task": tmp_dic['sub_task'],
            "progression_state": tmp_dic['current_state'],
            "cluster_index":tmp_dic['topk_cluster_index'],
            "labels": torch.tensor([tmp_dic['labels'], tmp_dic['item_id']]),
        }
        return ans

    def format_one_test_case(self, dialogue_history, topic_list, current_state, topk_cluster_index):
        ans_list = []
        for one_topic in topic_list:
            tmp_dic = {
                    "item_id": 0,
                    "history": dialogue_history,
                    "topic": one_topic,
                    "labels": 1,
                    "current_state":current_state,
                    "topk_cluster_index":topk_cluster_index
                }
            ans_list.append(tmp_dic)
        return ans_list

    def __getitem__(self, item):## item是编号， item一个整数， item>=0 item< __len()__
        tmp_dic = self.total_data[item]
        return self.get_model_input(tmp_dic)

    def __len__(self):
        # return 1004
        return len(self.total_data)

class TopicRankingDatasetForTest(Dataset):
    def __init__(self, dialogue_history, topic_list, current_state, distance_list, tokenizer, max_history_len=128, max_topic_len=16):
        super(TopicRankingDatasetForTest, self).__init__()
        self.tokenizer = tokenizer
        self.max_history_len = max_history_len
        self.max_topic_len = max_topic_len
        self.cluster_num = 15
        self.total_data = []
        topk_cluster_index = []
        for one_distance in distance_list:
            topk_cluster_index.append(np.argsort(one_distance)[-self.cluster_num:])

        for one_topic in topic_list:
            tmp_dic = {
                    "item_id": 0,
                    "history": dialogue_history,
                    "topic": one_topic,
                    "labels": 1,
                    "current_state":current_state,
                    "topk_cluster_index":topk_cluster_index
                }

            self.total_data.append(tmp_dic)
        # if 'test' in file_path:
        #     self.total_data = self.total_data[:1000]
    def encode_sentence(self, history, topic):
        history_token = self.tokenizer.encode(history, add_special_tokens=False)
        topic_token = self.tokenizer.encode(topic, add_special_tokens=False)
        input_ids = [self.tokenizer.cls_token_id] + history_token[-self.max_history_len:] + \
                    [self.tokenizer.sep_token_id] + topic_token[:self.max_topic_len] + [self.tokenizer.sep_token_id]
        return self.padding_sentence(input_ids)

    def padding_sentence(self, input_list):
        max_len = self.max_topic_len + self.max_history_len + 3
        return input_list + [self.tokenizer.pad_token_id] * max(max_len - len(input_list), 0)

    def get_str_form(self, tmp_str, tmp_len):
        str_list = tmp_str.split()[:tmp_len]
        return ' '.join(str_list)


    def get_model_input(self, tmp_dic):
        ans = {
            "input_ids": torch.tensor(self.encode_sentence(tmp_dic['history'], tmp_dic['topic'])),
            # "sub_task": tmp_dic['sub_task'],
            "progression_state": tmp_dic['current_state'],
            "cluster_index":tmp_dic['topk_cluster_index'],
            "labels": torch.tensor([tmp_dic['labels'], tmp_dic['item_id']]),
        }
        return ans

    def __getitem__(self, item):## item是编号， item一个整数， item>=0 item< __len()__
        tmp_dic = self.total_data[item]
        return self.get_model_input(tmp_dic)

    def __len__(self):
        # return 1004
        return len(self.total_data)