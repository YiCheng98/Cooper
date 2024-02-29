import sys
sys.path.append('..')
import os
import json
from utils.call_LLM_API.my_API_call import call_LLM
from utils.prompt_management.PromptLoader import PromptLoader
from utils.nlgeval import calc_nlg_metrics
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inference_data_path", type=str, default="../TopicRanking/inference_results/P4G/test.json")
parser.add_argument("--output_path", type=str, default="results/P4G.json")
parser.add_argument("--dataset_type", type=str, default="P4G")
parser.add_argument("--num_candidates", type=int, default=3)

args = parser.parse_args()
inference_data_path = args.inference_data_path
output_path = args.output_path
dataset_type = args.dataset_type
num_candidates = args.num_candidates

if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
prompt_loader = PromptLoader()

def count_turns(dialog_text):
    return dialog_text.count(": ")

def generate_esc_prompt_data():
    with open(inference_data_path, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
    prompt_template_tc = prompt_loader.load_prompt(f"{dataset_type}_generate_response_with_topic_candidates")
    prompt_template_pr = prompt_loader.load_prompt(f"{dataset_type}_generate_response_passive")
    results = []
    for sample in tqdm(data):
        dialogue_history = sample['dialogue_history']
        topic_candidates = sample['topic_candidates']
        ranking_result = sample['ranking_result']
        gold_response = sample['gold_response']
        selected_topics = [topic_candidates[r-1]['topic'] for r in ranking_result][:num_candidates]
        selected_topics_text = '\n'.join([f"{topic_index+1}. {topic}" for topic_index, topic in enumerate(selected_topics)])
        if count_turns(dialogue_history)>2:
            prompt = prompt_template_tc.format(dialogue_history=dialogue_history, topic_candidates=selected_topics_text)
        else:
            prompt = prompt_template_pr.format(dialogue_history=dialogue_history)
        output = call_LLM(prompt)
        results.append({'dialogue_history': dialogue_history,
                        'gold_response': gold_response,
                        'output': output})
    return results


if __name__ == "__main__":
    results = generate_esc_prompt_data()
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(results))
    print(f"Results are saved in {output_path}")
    preds = [result['output'] for result in results]
    labels = [result['gold_response'] for result in results]
    metrics = calc_nlg_metrics(preds, labels)
    print(f"Metrics: {metrics}")

