
Please first download the data from [Google Drive](https://drive.google.com/drive/folders/13J9wmRqAuXSSC7PtGyclTfVouD4doLSq?usp=sharing) and put them under the current directory. The link provides all the data we used in our experiments. The subdirectories named "esconv/" and "P4G/" contain our annotated data based on the original [ESConv](https://huggingface.co/datasets/thu-coai/esconv/tree/main) [\[1\]](#jump1) and [P4G](https://github.com/ohyj1002/persuasionforgood/tree/master/data) [\[2\]](#jump2) datasets. 

## Original Data
The subdirectory "**./*/original_data/**" provides the original data of ESConv and P4G datasets. For ESConv, we used the preprocessed version from [Cheng et al.](https://github.com/lwgkzl/MultiESC/tree/main/MultiESC/data) [\[3\]](#jump3).

## Annotation with ChatGPT
For each generation turn from the system side, we automatically annotated the **state summaries** of each dialogue goal aspect and the potential **topic candidates** at that turn, using GPT-3.5-turbo. The subdirectory "**./*/api_annotated/**" provides the annotated_data. All the annotations are saved under the **"states"** key of each dialogue turn. 

- *Dialogue Goal Aspect*: We argue that complex dialogue goals can be typically divided into several inter-connected aspects. For instance, Emotional Support Conversations (ESC) should include three key aspects: exploration, comforting, and action. Please refer to the Preliminaries section in our paper for details about how we define the dialogue goal aspects on our experimental tasks. 
- *State Summary* : It aims to summarize the previous efforts in achieving the given dialogue goal aspect. For example, to get the state summary for the exploration aspect in ESC, we prompt the LLM to “*summarize the seeker’s experience that caused their emotional distress*.”
- *Topic Candidate*: It can be seen as a brief content outline for the following utterance. For instance, the aspect promoter of the exploration agent in ESC is implemented by instructing an LLM to “*list three questions that the supporter can ask the seeker to further understand their situation (each less than 20 words)*”. The number of topic candidates for each turn may vary between 8~12 due to instability of LLM annotation. 

Please refer to the appendix of our paper for all the prompt templates we used for annotation. 

## Pseudo-Labeling for Topic Ranking Results
We conducted pseudo-labeling for the topic ranking results, in order to provide supervision to train the topic candidate ranker in our framework. 

We use [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) to measure text similarity during this labeling process. Please first install sentence-transformers so that you could load the model easily:
```
pip install -U sentence-transformers
```

Run the following commands to conduct pseudo-labeling on the two datasets. If, for some reason, you could not automatically download the model by running the following commands, please manually download it by referring to this [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
```
python psuedo_label_topic_ranking.py --mode esconv
python psuedo_label_topic_ranking.py --mode P4G
```



The labeling process will typically take 2~3 hours on each dataset. The labeling results will be saved under the  "**./*/api_annotated_w_ranking/**" subdirectory. The ranking scores and labels will be saved under the **"ranking scores"** key. 

Note that the original P4G dataset only provides strategy annotation on part of the samples (300 dialogue samples, specifically). Since our pseudo-labeling relies on that strategy annotation, we only conduct pseudo-labeling on these samples. When splitting the dataset, we ensure all the samples in the validation and test sets are annotated with strategies and pseudo-labeling results for topic ranking.

## References
<span id="jump1">[1]</span> Liu, Siyang, et al. "Towards Emotional Support Dialog Systems." _ACL_. 2021.

<span id="jump2">[2]</span> Wang, Xuewei, et al. "Persuasion for Good: Towards a Personalized Persuasive Dialogue System for Social Good." _ACL_. 2019.

<span id="jump3">[3]</span> Cheng, Yi, et al. "Improving Multi-turn Emotional Support Dialogue Generation with Lookahead Strategy Planning." _EMNLP_. 2022.
