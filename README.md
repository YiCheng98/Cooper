# Cooper
This repository will provide the codes and data used in the AAAI'24 paper, [*COOPER: Coordinating Specialized Agents towards a Complex Dialogue Goal*](https://arxiv.org/pdf/2312.11792.pdf).

## Data Preparation
The ``data/`` directory provides the annotated data we used in our experiments. Please refer to the ``data/README.md`` file for details about the downloading address and how to use the preprocessing codes.

## Topic Ranking
The codes for implementing the topic ranking module in provided in the ``TopicRanking`` directory. Please run the following commands for obtaining the *typical target states*  on the ESConv dataset (please refer to the *Local Analysis with Specialized Agents* for detailed explanation), and prepare the data for training the topic ranking module.
```
cd TopicRanking
python subgoal_state_clustering.py --dataset_type esconv
python generate_ranking_data.py --dataset_type esconv
```
Run the following command to train the topic ranking module.
```
python train.py --data_dir data/esconv/ --output_dir output/esconv/
```
Run the following command to conduct inference on the test set.
```
python inference.py \
--model_path output/esconv/ \
--data_dir data/esconv/ \
--output_dir inference_results/esconv/
```
To conduct the experiment on the P4G dataset, you can substitute ``esconv`` with ``P4G`` in the above commands.


## Utterance Generator

### Preparation
To implement the prompt-based generator, you need to first set up the OpenAI API key. Specifically, you can simply do this by input the key in the ``utils/call_LLM_API/api_key.txt`` file.

To calculate the NLG metrics for evaluation, please download the evaluation codes ``nlgeval/`` from this [link](https://drive.google.com/file/d/1SjKkmuP5xo1Pfsfpjup61L1_UYKXiNFz/view?usp=sharing). Decompress the zip file and put it in the ``utils/`` directory.

### Prompt-based Generator
Please run the following command to use the prompt-based generator to generate the utterances on the ESConv dataset.
``` 
cd PT_Generator
python inference.py --inference_data_path ../TopicRanking/inference_results/esconv/test.json --output_path results/esconv.json --dataset_type esconv
```
To conduct the experiment on the P4G dataset, you can substitute all ``esconv`` with ``P4G`` in the above command.

### Finetuned Generator
To train the finetuned generator, please run the following command.
```
cd FT_Generator
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--model_path facebook/bart-base \
--output_dir ./model/esconv/  \
--data_dir ../TopicRanking/data/esconv/
```
Tp conduct inference on the test set, please run the following command.
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--do_train False \
--model_path ./model/esconv/ \
--data_dir ../TopicRanking/data/esconv/
```
To conduct the experiment on the P4G dataset, you can substitute all ``esconv`` with ``P4G`` in the above command.

