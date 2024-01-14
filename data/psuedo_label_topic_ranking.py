import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os

SENTENCE_ENCODER = "all-mpnet-base-v2"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./", help="directory of the data")
parser.add_argument("--mode", type=str, default="esconv", help="processed_data can either be 'esconv' or 'P4G'")
args = parser.parse_args()
data_dir = args.data_dir
mode = args.mode
if args.mode not in ["esconv", "P4G"]:
    raise Exception("processed_data can only be 'esconv' or 'P4G'")
dataset_path = os.path.join(data_dir, mode)

class TopicRankingPseudoLabeller_esconv():
    def __init__(self, data):
        self.data = data

    def map_strategy2subgoal(self, strategy):
        if strategy == "Question":
            return "Exploration"
        if strategy == "Providing Suggestions or Information":
            return "Action"
        if strategy == "Passive Response":
            return "Passive Response"
        return "Comfort"

    def calculate_text_similarity(self, model, reference, text):
        embedding1 = model.encode([reference])
        embedding2 = model.encode([text])

        # normalize
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)

        embedding1_norm = embedding1_norm.reshape(1, -1)
        embedding2_norm = embedding2_norm.reshape(1, -1)

        similarity = cosine_similarity(embedding1_norm, embedding2_norm)[0][0]

        return float(similarity)

    def calculate_summary_scores(self, ranking_scores):
        scores = [ranking_scores["subgoal_acc"], ranking_scores["strategy_acc"], ranking_scores["text_similarity"]]
        weights = [100, 10, 1]
        summary_score = 0
        for i in range(3):
            summary_score += scores[i]*weights[i]
        return summary_score

    def rank_topic_candidates(self, topic_candidates):
        sorted_lst = sorted(topic_candidates, key=lambda x: self.calculate_summary_scores(x['ranking_scores']), reverse=True)
        for i, item in enumerate(sorted_lst):
            item['ranking_scores']["ranking"] = i+1
        return sorted_lst

    def annotate_data(self):
        sentence_embedding_model = SentenceTransformer(SENTENCE_ENCODER)
        for sample in tqdm(self.data):
            for turn in sample["dialog"]:
                if "states" not in turn.keys():
                    continue
                topic_candidates = turn["states"]["topic_candidates"]
                response = turn["text"]
                all_strategy = turn["all_strategy"]

                #Add the scores of each candidate, including the calculation of "subgoal_acc", "strategy_acc", "text_similarity", and "ranking" (ranked comprehensively based on the three former scores).
                gold_subgoals = set([self.map_strategy2subgoal(strategy) for strategy in all_strategy])
                for topic_candidate in topic_candidates:
                    topic_candidate["ranking_scores"] = {
                        "subgoal_acc": 0,
                        "strategy_acc": 0,
                        "text_similarity": 0,
                        "ranking": 0
                    }
                    ranking_scores = topic_candidate["ranking_scores"]
                    subgoal, strategy, topic = topic_candidate["subgoal"], topic_candidate["strategy"], topic_candidate["topic"]
                    if subgoal in gold_subgoals:
                        ranking_scores["subgoal_acc"] = 1
                    if strategy in all_strategy:
                        ranking_scores["strategy_acc"] = 1
                    if subgoal == "Passive Response" and ranking_scores["subgoal_acc"] == 1:
                        ranking_scores["text_similarity"] = 1
                    else:
                        ranking_scores["text_similarity"] = self.calculate_text_similarity(sentence_embedding_model, response, topic)

                topic_candidates = self.rank_topic_candidates(topic_candidates)
                turn["states"]["topic_candidates"] = topic_candidates

class TopicRankingPseudoLabeller_P4G():
    def __init__(self, data):
        self.data = data
        strategy_mapping = dict()
        subgoal_mapping = dict()
        with open(f"{dataset_path}/strategy_mapping.txt", 'r', encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            tmp = line.split(', ')
            strategy_mapping[tmp[0]] = tmp[1]
            subgoal_mapping[tmp[0]] = tmp[2][:-1]
        self.strategy_mapping = strategy_mapping
        self.subgoal_mapping = subgoal_mapping

    def calculate_text_similarity(self, model, reference, text):
        embedding1 = model.encode([reference])
        embedding2 = model.encode([text])

        # normalize
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)

        embedding1_norm = embedding1_norm.reshape(1, -1)
        embedding2_norm = embedding2_norm.reshape(1, -1)

        similarity = cosine_similarity(embedding1_norm, embedding2_norm)[0][0]

        return float(similarity)

    def calculate_summary_scores(self, ranking_scores):
        scores = [ranking_scores["subgoal_acc"], ranking_scores["strategy_acc"], ranking_scores["text_similarity"]]
        weights = [100, 10, 1]
        summary_score = 0
        for i in range(3):
            summary_score += scores[i] * weights[i]
        return summary_score

    def rank_topic_candidates(self, topic_candidates):
        sorted_lst = sorted(topic_candidates, key=lambda x: self.calculate_summary_scores(x['ranking_scores']), reverse=True)
        for i, item in enumerate(sorted_lst):
            item['ranking_scores']["ranking"] = i+1
        return sorted_lst

    def get_subgoal_score(self, pred_subgoal, ref_strategy):
        ref_strategy = list(set(ref_strategy))
        if 'other' in ref_strategy and len(ref_strategy)>1:
            ref_strategy = [i for i in ref_strategy if i!='other']
        ref_subgoals = []
        for s in ref_strategy:
            tmp = self.subgoal_mapping[s.lower()]
            if tmp not in ref_subgoals:
                ref_subgoals.append(tmp)
        pred_subgoal = pred_subgoal.lower()
        if pred_subgoal in ref_subgoals:
            return 1
        else:
            return 0

    def get_strategy_score(self, pred_strategy, ref_strategy):
        ref_strategy = list(set(ref_strategy))
        if 'other' in ref_strategy and len(ref_strategy) > 1:
            ref_strategy = [i for i in ref_strategy if i != 'other']
        ref_strategy_processed = []
        for s in ref_strategy:
            tmp = self.strategy_mapping[s.lower()]
            if tmp not in ref_strategy_processed:
                ref_strategy_processed.append(tmp)
        pred_strategy = pred_strategy.lower()
        if pred_strategy in ref_strategy_processed:
            return 1
        else:
            return 0

    def annotate_data(self):
        sentence_embedding_model = SentenceTransformer(SENTENCE_ENCODER)
        for sample in tqdm(self.data):
            for turn in sample["dialog"]:
                if "states" not in turn.keys() or turn["strategy"] == []:
                    continue
                topic_candidates = turn["states"]["topic_candidates"]
                response = turn["text"]

                for topic_candidate in topic_candidates:
                    topic_candidate["ranking_scores"] = {
                        "subgoal_acc": 0,
                        "strategy_acc": 0,
                        "text_similarity": 0,
                        "ranking": 0
                    }
                    ranking_scores = topic_candidate["ranking_scores"]
                    ranking_scores["subgoal_acc"] = self.get_subgoal_score(topic_candidate['subgoal'], turn['strategy'])
                    ranking_scores["strategy_acc"] = self.get_strategy_score(topic_candidate['strategy'],
                                                                             turn['strategy'])
                    ranking_scores["text_similarity"] = self.calculate_text_similarity(sentence_embedding_model, response, topic_candidate['topic'])

                topic_candidates = self.rank_topic_candidates(topic_candidates)
                turn["states"]["topic_candidates"] = topic_candidates

if __name__ == '__main__':
    file_names = []
    for root, dirs, files in os.walk(f"{dataset_path}/api_annotated"):
        for file in files:
            if file.endswith(".json"):
                file_names.append(os.path.join(root, file))

    for src_file in file_names:
        trg_file = src_file.replace(f"{dataset_path}/api_annotated", f"{dataset_path}/api_annotated_w_ranking")
        print(trg_file)
        # if os.path.exists(trg_file):
        #     continue
        data = []
        with open(src_file, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        if mode == "esconv":
            annotator = TopicRankingPseudoLabeller_esconv(data)
        else:
            annotator = TopicRankingPseudoLabeller_P4G(data)
        annotator.annotate_data()
        with open(trg_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(annotator.data))