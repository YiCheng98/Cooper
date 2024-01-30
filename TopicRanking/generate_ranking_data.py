import numpy as np
import json
from subgoal_state_clustering import get_sentence_embedding
from tqdm import tqdm
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, default="P4G", help="either be 'esconv' or 'P4G'")
parser.add_argument("--skip_exist", type=bool, default=False, help="whether to skip existing files")
args = parser.parse_args()
dataset_type = args.dataset_type

cluster_info_file = f"data/{dataset_type}/cluster_info.json"

K_REMAIN = 3
speaker_sep_token = " "
data_dir = f"../data/{dataset_type}/api_annotated_w_ranking/"
cluster_info_dir = f"./data/{dataset_type}/"
output_dir = f"./data/{dataset_type}/"

if dataset_type=="P4G":
    state_keys = ["inquiry_state", "appeal_state", "proposition_state"]
elif dataset_type=="esconv":
    state_keys = ["exploration_state", "comfort_state", "action_state"]
else:
    raise Exception("dataset_type can only be 'esconv' or 'P4G'")

def transfrom2rankingdata(dialog_data):
    ranking_data = []
    for sample in dialog_data:
        for i, turn in enumerate(sample["dialog"]):
            if "states" not in turn.keys():
                continue
            tmp = sample["dialog"][:i]#[max(0, i-4):i]
            dialogue_history = speaker_sep_token.join([f"{item['speaker']}: {item['text']}" for item in tmp])
            ranking_data.append({"dialogue_history": dialogue_history,
                                 "topic_candidates": turn["states"]["topic_candidates"],
                                 "progression_info":  turn["states"]["progression_info"]})
    return ranking_data

def calculate_distances(centers, v):
    distances = np.sum((centers - v)**2, axis=1)
    return distances.tolist()



if __name__ == '__main__':

    with open(cluster_info_file, "r", encoding="utf-8") as f:
        cluster_info = json.loads(f.read())
    centers = dict()
    clusterers = dict()
    for info in cluster_info:
        state_key = info["state_key"]
        centers[state_key] = np.load(os.path.join(cluster_info_dir, f'{state_key}_centers.npy'))

    for f_name in os.listdir(data_dir):
        if dataset_type=="P4G" and f_name=="train.json":
            continue
        data = []
        file_path = os.path.join(data_dir, f_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data += json.loads(f.read())
        for i in tqdm(range(len(data)), desc=f"Transforming data in {file_path}"):
            for j, turn in enumerate(data[i]['dialog']):
                if "states" not in turn.keys():
                    continue
                states = turn["states"]
                progression_info = dict()
                for state_key in state_keys:
                    state = states[state_key]
                    state_embedding = get_sentence_embedding(state).tolist()
                    distances = calculate_distances(centers[state_key], state_embedding)
                    progression_info[state_key] = {"state_embedding": state_embedding, "distances": distances}
                data[i]["dialog"][j]["states"]["progression_info"] = progression_info
        with open(os.path.join(output_dir, f_name), "w", encoding="utf-8") as f:
            f.write(json.dumps(data))
        if dataset_type=="P4G" and f_name=="train_annotated.json":
            f_name = "train.json"
        ranking_data = transfrom2rankingdata(data)
        with open(f"{output_dir}/{f_name}", "w", encoding="utf-8") as f:
            f.write(json.dumps(ranking_data))


