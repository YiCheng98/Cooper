import sys
import json
sys.path.append('..')
from utils.prompt_management.PromptLoader import PromptLoader
import torch
def count_turns(dialog_text):
    return dialog_text.count(": ")

def recent_dialog(dialog_text, consider_turn_num=10):
    turns = dialog_text.split(": ")
    turns = turns[-consider_turn_num:]
    return ": ".join(turns)

class Dataset:
    def __init__(self, tokenizer, file_path, num_candidates, max_source_len=512, max_target_len=128):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.loads(f.read())
        results = []
        if tokenizer.sep_token_id is None:
            sep_token = ' '
        else:
            sep_token = tokenizer.sep_token
        for sample in data:
            dialogue_history = recent_dialog(sample['dialogue_history']) + " sys: "
            states = []
            if "states" in sample.keys():
                for state_key in sample["states"].keys():
                    states.append(sample['states'][state_key])
            topic_candidates = sample['topic_candidates']
            # ranking_result = sample['ranking_result']
            gold_response = sample['gold_response']
            selected_topics = [topic['topic'] for topic in topic_candidates][:num_candidates]
            input = sep_token.join(selected_topics + states + [dialogue_history])
            results.append({'dialogue_history': dialogue_history,
                            'gold_response': gold_response,
                            'input': input})
        self.total_data = results

    def get_model_input(self, tmp_dic):
        input_ids = self.tokenizer.encode(tmp_dic['input'], add_special_tokens=False)
        # truncate the input_ids to max_source_len
        input_ids = input_ids[:self.max_source_len]
        # pad the input_ids to max_source_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * max(self.max_source_len - len(input_ids), 0)
        labels = self.tokenizer.encode(tmp_dic['gold_response'], add_special_tokens=False)
        labels = labels[:self.max_target_len]
        labels = labels + [self.tokenizer.pad_token_id] * max(self.max_target_len - len(labels), 0)
        ans = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }
        return ans

    def __getitem__(self, item):  ## item>=0 item< __len()__
        tmp_dic = self.total_data[item]
        return self.get_model_input(tmp_dic)

    def __len__(self):
        return len(self.total_data)