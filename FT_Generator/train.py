import sys
sys.path.append('..')

import json
import os

from utils.nlgeval import calc_nlg_metrics
from DataReader import Dataset
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer
from transformers.optimization import AdamW
import argparse
import logging
import os
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, Seq2SeqTrainer
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", default="lcsts",type=str)
parser.add_argument("--lr2",default=1e-4,type=float)
parser.add_argument("--do_train",default=True)
parser.add_argument("--do_eval",default=True)
parser.add_argument("--do_predict",default=True)
parser.add_argument("--per_device_train_batch_size", default=32, type=int)
parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--warmup_ratio", default=0.0, type=float)
parser.add_argument("--max_source_length", default=512, type=int)
parser.add_argument("--generation_max_length", default=64, type=int)
parser.add_argument("--seed", default=3407, type=int)
parser.add_argument("--save_total_limit", default=3, type=int)
parser.add_argument("--num_train_epochs", default=40, type=int)
parser.add_argument("--metric_for_best_model", default="Bleu_2",type=str)
parser.add_argument("--greater_is_better", default=True)
parser.add_argument("--evaluation_strategy", default="epoch",type=str)
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--save_strategy", default="epoch", type=str)
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--ignore_pad_token_for_loss", default=True)
parser.add_argument("--predict_with_generate", default=True)
parser.add_argument("--num_beams", default=1, type=int)
parser.add_argument("--not_pretrain", action="store_true")
parser.add_argument("--repetition_penalty", default=1.0, type=float)
# parser.add_argument("--config_path", default='../../MODEL/transformer_config', type=str)
parser.add_argument("--dataset_type", type=str, default="esconv", help="either be 'esconv' or 'P4G'")
parser.add_argument("--num_candidates", type=int, default=3, help="number of candidates")
parser.add_argument("--model_path", type=str, default="facebook/bart-base", help="model path")
parser.add_argument("--output_dir", type=str, default="./model/esconv/", help="output directory")
parser.add_argument("--data_dir", type=str, default="../TopicRanking/inference_results/esconv/", help="data directory")
# parser.add_argument("--max_eval_samples", type=int, default=200, help="max evaluation samples")

args = parser.parse_args()
arg_dict = args.__dict__
print("args:", arg_dict)
logger = logging.getLogger(__name__)

args = parser.parse_args()
train_parser = HfArgumentParser(Seq2SeqTrainingArguments)

dataset_type = args.dataset_type
num_candidates = args.num_candidates
model_path = args.model_path
# max_eval_samples = args.max_eval_samples
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


model, loading_info = BartForConditionalGeneration.from_pretrained(model_path, output_loading_info=True)
tokenizer = BartTokenizer.from_pretrained(model_path)

train_file = args.data_dir + "train.json"
dev_file = args.data_dir + "dev.json"
test_file = args.data_dir + "test.json"

train_set = Dataset(tokenizer, train_file, num_candidates)
dev_set = Dataset(tokenizer, dev_file, num_candidates)
test_set = Dataset(tokenizer, test_file, num_candidates)
# dev_set.total_data = dev_set.total_data[:max_eval_samples]

def compute_metrics(eval_preds):
    decoder_preds, decoder_labels = eval_preds
    ref_list = []
    hyp_list = []
    for ref, hyp in zip(decoder_labels, decoder_preds):
        ref = tokenizer.decode(ref, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        hyp = [tokenizer.pad_token_id if index<0 else index for index in hyp]
        hyp = tokenizer.decode(hyp, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if len(hyp) == 0:
            hyp = '&'
        ref_list.append(ref)
        hyp_list.append(hyp)
    # print 10 examples
    for i in range(3):
        print("ref:", ref_list[i])
        print("hyp:", hyp_list[i])
        print()
    return calc_nlg_metrics(decoder_preds=hyp_list, decoder_labels=ref_list)
def get_optimizer(model, second_parameter):
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
    optimizer = AdamW(optimizer_grouped_parameters)
    return optimizer

if __name__== "__main__":
    # define the optimizer (AdamW)
    second_parameter = ["model.encoder.embed_positions.weight", "model.decoder.embed_positions.weight"]
    optimizer = get_optimizer(model, second_parameter)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        overwrite_output_dir=args.overwrite_output_dir,
        warmup_ratio=args.warmup_ratio,
        generation_max_length=args.generation_max_length,
        seed=args.seed,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        predict_with_generate=args.predict_with_generate,
        generation_num_beams=args.num_beams,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        optimizers=(optimizer, None)
    )
    if not args.do_train=='False':
        trainer.train()
        # save the model
        model.save_pretrained(output_dir)
    if not args.do_predict=='False':
        outputs = trainer.predict(test_set)
        labels = outputs.label_ids
        preds = outputs.predictions
        results = test_set.total_data
        for i, sample in enumerate(results):
            pred = preds[i]
            pred = [tokenizer.pad_token_id if index<0 else index for index in pred]
            output = tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sample['output'] = output
        if not os.path.exists("results"):
            os.makedirs("results")
        with open(f"results/{dataset_type}.json", 'w', encoding="utf-8") as f:
            f.write(json.dumps(results))
        print(f"Results are saved in results/{dataset_type}.json")
        metrics = outputs.metrics
        print(f"Metrics on the test set: {metrics}")