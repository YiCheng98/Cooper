# Cooper
This repository will provide the codes and data used in the AAAI'24 paper, [*COOPER: Coordinating Specialized Agents towards a Complex Dialogue Goal*](https://arxiv.org/pdf/2312.11792.pdf).

We have already uploaded the data used in our work and the codes for topic ranking. The remaining codes will be released soon, before March 1st, 2023.

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
python train.py --data_dir data/esconv/ --output_dir esconv
```
To conduct the experiment on the P4G dataset, you can substitute ``--dataset_type esconv`` with ``--dataset_type P4G``.
