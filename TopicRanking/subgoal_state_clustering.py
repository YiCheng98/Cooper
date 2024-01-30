from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import os
from sklearn.metrics import silhouette_score
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, default="P4G", help="either be 'esconv' or 'P4G'")
parser.add_argument("--skip_exist", type=bool, default=True, help="whether to skip existing files")
parser.add_argument("--search_best_cluster_num", type=bool, default=True, help="whether to skip existing files")
args = parser.parse_args()
dataset_type = args.dataset_type

data_dir = f"../data/{dataset_type}/api_annotated_w_ranking/"
output_dir = f"./data/{dataset_type}/"
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)
training_data_file = os.path.join(data_dir, "train.json")
val_data_file = os.path.join(data_dir, "dev.json")
cluster_info_file = os.path.join(output_dir, "cluster_info.json")

if dataset_type=="P4G":
    state_keys = ["inquiry_state", "appeal_state", "proposition_state"]
elif dataset_type=="esconv":
    state_keys = ["exploration_state", "comfort_state", "action_state"]
else:
    raise Exception("dataset_type can only be 'esconv' or 'P4G'")

skip_exist = args.skip_exist

LOAD_CACHE = skip_exist
SEARCH_CLUSTER_NUM = args.search_best_cluster_num

def get_sentence_embedding(text, model=SentenceTransformer('all-mpnet-base-v2')):
    return model.encode([text])

def get_states(state_key, file_name=training_data_file):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    with open(val_data_file, "r", encoding="utf-8") as f:
        data += json.loads(f.read())
    states = []
    for sample in data:
        dialogue = sample['dialog']
        final_sys_turn = dialogue[-1]
        if dataset_type=="P4G" and dialogue[-1]['speaker'] != 'Persuader':
            final_sys_turn = dialogue[-2]
        if dataset_type=="esconv" and dialogue[-1]['speaker'] != 'sys':
            final_sys_turn = dialogue[-2]
        state = final_sys_turn["states"][state_key]
        if len(state)>0:
            states.append(state)
    return states

def cluster(exploration_states, embeddings_file, cluster_num=0):
    if LOAD_CACHE and os.path.exists(embeddings_file):
        exploration_state_embeddings = np.load(embeddings_file)
    else:
        exploration_state_embeddings = np.array(
            [get_sentence_embedding(state).reshape(1, -1)[0] for state in tqdm(exploration_states, desc="Calculating embeddings")])
        np.save(embeddings_file, exploration_state_embeddings)

    data = exploration_state_embeddings

    if SEARCH_CLUSTER_NUM or cluster_num<=0:
        min_cluster_num = int(len(data)/50)
        max_cluster_num = int(len(data)/20)
        print(f"Searching for best cluster num in [{min_cluster_num}, {max_cluster_num})")
        best_silhouette_score = 0
        best_cluster_num = 10
        for i in tqdm(range(min_cluster_num, max_cluster_num), desc="Searching for best cluster num "):
            clusterer = KMeans(n_clusters=i, n_init="auto")
            clusterer.fit(data)
            labels = clusterer.labels_.tolist()
            score = silhouette_score(data, labels)
            if score > best_silhouette_score:
                best_cluster_num = i
                best_silhouette_score = score
        cluster_num = best_cluster_num

    print(f"CLUSTER_NUM = {cluster_num}")#, silhouette_score = {best_silhouette_score}")
    clusterer = KMeans(n_clusters=cluster_num, n_init="auto")
    clusterer.fit(data)
    labels = clusterer.labels_.tolist()
    clustered_data = [[] for i in range(cluster_num)]
    for tuple in zip(exploration_states, labels):
        state, label = tuple
        clustered_data[label].append(state)


    return clustered_data, cluster_num, clusterer

def create_P4G_train_annotated():
    train_annotated_data_path = os.path.join(data_dir, "train_annotated.json")
    if os.path.exists(train_annotated_data_path) is False:
        with open(os.path.join(data_dir, "train.json"), "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        train_annoated_data = []
        for sample in data:
            dialog = sample["dialog"]
            turn = dialog[-1]
            if turn["speaker"] != "Persuader":
                turn = dialog[-2]
            if len(turn['strategy']) == 0:
                continue
            else:
                train_annoated_data.append(sample)
        with open(os.path.join(data_dir, "train_annotated.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(train_annoated_data))

if __name__ == '__main__':
    # create train_annotated.json for P4G dataset
    if dataset_type == "P4G":
        create_P4G_train_annotated()
        training_data_file = os.path.join(data_dir, "train_annotated.json")

    if os.path.exists(cluster_info_file) and not SEARCH_CLUSTER_NUM:
        with open(cluster_info_file, "r", encoding="utf-8") as f:
            cluster_info = json.loads(f.read())
    else:
        SEARCH_CLUSTER_NUM = True
        cluster_info = [{"state_key": state_keys[i], "cluster_num": 0} for i in range(3)]
    for i, info in enumerate(cluster_info):
        state_key = info["state_key"]
        print(f"=============CLUSTERING {info['state_key']}=============")
        output_path = os.path.join(output_dir, f'{state_key}_centers.npy')
        # if os.path.exists(output_path) and skip_exist:
        #     print(f"Skip {info['state_key']} because {output_path} exists")
        #     continue
        cluster_num = info["cluster_num"]
        states = get_states(state_key)
        clustered_data, cluster_num, clusterer = cluster(states, os.path.join(output_dir, f"{state_key}_embeddings.npy"), cluster_num=cluster_num)
        cluster_info[i]["cluster_num"] = cluster_num
        # with open(os.path.join(output_dir, f'/{state_key}_cluster.pkl'), 'wb') as file:
        #     pickle.dump(clusterer, file)
        np.save(output_path, clusterer.cluster_centers_)

    if SEARCH_CLUSTER_NUM:
        with open(cluster_info_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(cluster_info))

