# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
from __future__ import print_function

import os
import warnings
import six
from six.moves import map

from myMetrics import Metric as MyMetric
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import nltk
import json

# str/unicode stripping in Python 2 and 3 instead of `str.strip`.
def _strip(s):
    return s.strip()


def compute_metrics(hypothesis, references, no_overlap=False, no_glove=False):
    with open(hypothesis, 'r') as f:
        hyp_list = f.readlines()
    ref_list = []
    for iidx, reference in enumerate(references):
        with open(reference, 'r') as f:
            ref_list.append(f.readlines())
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        scorers = [
            # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.6f" % (m, sc))
                    ret_scores[m] = sc
            else:
                print("%s: %0.6f" % (method, score))
                ret_scores[method] = score
            if isinstance(scorer, Meteor):
                scorer.close()
        del scorers

    if not no_glove:
        from metric.word2vec.evaluate import eval_emb_metrics
        import numpy as np

        glove_hyps = [h.strip() for h in hyp_list]
        ref_list_T = np.array(ref_list).T.tolist()
        glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
        scores, scores_list_dict = eval_emb_metrics(glove_hyps, glove_refs)
        print(scores)
        scores = scores.split('\n')
        for score in scores:
            name, value = score.split(':')
            value = float(value.strip())
            ret_scores[name] = value

    return ret_scores


def compute_individual_metrics(ref, hyp, no_overlap=False, no_glove=False):
    assert isinstance(hyp, six.string_types)

    if isinstance(ref, six.string_types):
        ref = ref.split('||<|>||')  # special delimiter for backward compatibility
    ref = [a.strip() for a in ref]
    refs = {0: ref}
    ref_list = [ref]

    hyps = {0: [hyp.strip()]}
    hyp_list = [hyp]

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc
            else:
                ret_scores[method] = score
            if isinstance(scorer, Meteor):
                scorer.close()
        del scorers

    if not no_glove:
        from metric.word2vec.evaluate import eval_emb_metrics
        import numpy as np

        glove_hyps = [h.strip() for h in hyp_list]
        ref_list_T = np.array(ref_list).T.tolist()
        glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
        scores, scores_list_dict = eval_emb_metrics(glove_hyps, glove_refs)
        scores = scores.split('\n')
        for score in scores:
            name, value = score.split(':')
            value = float(value.strip())
            ret_scores[name] = value

    return ret_scores


class NLGEval(object):
    glove_metrics = {
        'EmbeddingAverageCosineSimilarity',
        'VectorExtremaCosineSimilarity',
        'GreedyMatchingScore',
    }

    valid_metrics = {
                        # Overlap
                        'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
                        'Distinct_1', 'Distinct_2', 'Distinct_3', 'Distinct_4',
                        'METEOR',
                        'ROUGE_L',
                        'CIDEr',

                        # Skip-thought
                        'SkipThoughtCS',
                    } | glove_metrics

    def __init__(self, no_overlap=False, no_glove=False,
                 metrics_to_omit=None):
        """
        :param no_overlap: Default: Use overlap metrics.
            `True` if these metrics should not be used.
        :type no_overlap: bool
        :param no_glove: Default: Use GloVe based metrics.
            `True` if these metrics should not be used.
        :type no_glove: bool
        :param metrics_to_omit: Default: Use all metrics. See `NLGEval.valid_metrics` for all metrics.
            The previous parameters will override metrics in this one if they are set.
            Metrics to omit. Omitting Bleu_{i} will omit Bleu_{j} for j>=i.
        :type metrics_to_omit: Optional[Collection[str]]
        """

        if metrics_to_omit is None:
            self.metrics_to_omit = set()
        else:
            self.metrics_to_omit = set(metrics_to_omit)
            # For backwards compatibility.
            if 'EmbeddingAverageCosineSimilairty' in self.metrics_to_omit:
                self.metrics_to_omit.remove('EmbeddingAverageCosineSimilairty')
                self.metrics_to_omit.add('EmbeddingAverageCosineSimilarity')

        assert len(self.metrics_to_omit - self.valid_metrics) == 0, \
            "Invalid metrics to omit: {}".format(self.metrics_to_omit - self.valid_metrics)

        self.no_overlap = no_overlap
        if not no_overlap:
            self.load_scorers()

        self.no_glove = no_glove or len(self.glove_metrics - self.metrics_to_omit) == 0
        if not self.no_glove:
            self.load_glove()

    def load_scorers(self):
        self.scorers = []

        omit_bleu_i = False
        for i in range(1, 4 + 1):
            if 'Bleu_{}'.format(i) in self.metrics_to_omit:
                omit_bleu_i = True
                if i > 1:
                    self.scorers.append((Bleu(i - 1), ['Bleu_{}'.format(j) for j in range(1, i)]))
                break
        if not omit_bleu_i:
            self.scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))

        if 'METEOR' not in self.metrics_to_omit:
            self.scorers.append((Meteor(), "METEOR"))
        if 'ROUGE_L' not in self.metrics_to_omit:
            self.scorers.append((Rouge(), "ROUGE_L"))
        if 'CIDEr' not in self.metrics_to_omit:
            self.scorers.append((Cider(), "CIDEr"))

    def load_glove(self):
        from word2vec.evaluate import Embedding
        from word2vec.evaluate import eval_emb_metrics
        import numpy as np
        self.eval_emb_metrics = eval_emb_metrics
        self.np = np
        self.glove_emb = Embedding()

    def compute_individual_metrics(self, ref, hyp):
        assert isinstance(hyp, six.string_types)
        ref = [a.strip() for a in ref]
        refs = {0: ref}
        ref_list = [ref]

        hyps = {0: [hyp.strip()]}
        hyp_list = [hyp]

        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, scs, m in zip(score, scores, method):
                        ret_scores[m] = sc
                else:
                    ret_scores[method] = score

        if not self.no_glove:
            glove_hyps = [h.strip() for h in hyp_list]
            ref_list_T = self.np.array(ref_list).T.tolist()
            glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
            scores, scores_list_dict = self.eval_emb_metrics(glove_hyps, glove_refs, emb=self.glove_emb,
                                           metrics_to_omit=self.metrics_to_omit)
            scores = scores.split('\n')
            for score in scores:
                name, value = score.split(':')
                value = float(value.strip())
                ret_scores[name] = value

        return ret_scores

    def compute_metrics(self, ref_list, hyp_list):
        ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
        refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
        assert len(refs) == len(hyps)
        
        ret_score_list = {}
        ret_scores = {}
        if not self.no_overlap:
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(refs, hyps)
                if isinstance(method, list):
                    for sc, scs, m in zip(score, scores, method):
                        ret_scores[m] = sc
                        ret_score_list[m] = [float(each) for each in scs]
                else:
                    ret_scores[method] = score
                    ret_score_list[method] = [float(each) for each in scores]

        if not self.no_glove:
            glove_hyps = [h.strip() for h in hyp_list]
            ref_list_T = self.np.array(ref_list).T.tolist()
            glove_refs = map(lambda refl: [r.strip() for r in refl], ref_list_T)
            scores, scores_list_dict = self.eval_emb_metrics(glove_hyps, glove_refs, emb=self.glove_emb)
            scores = scores.split('\n')
            for score in scores:
                name, value = score.split(':')
                value = float(value.strip())
                ret_scores[name] = value
            ret_score_list.update(scores_list_dict)
        
        return ret_scores, ret_score_list

def calc_distinct_k(hyps, k):
    d = {}
    tot = 0
    for sen in hyps:
        tokens = nltk.word_tokenize(sen.lower())
        for i in range(0, len(tokens)-k+1):
            key = tuple(tokens[i:i+k])
            d[key] = 1
            tot += 1
    if tot > 0:
        dist = len(d) / tot
    else:
        warnings.warn('the distinct is invalid')
        dist = 0.
    return dist

def get_P4G_sota_data():
    dir_path = "../P4G_sota"
    file_names = os.listdir(dir_path)
    ref_list = []
    hyp_lists = {}
    for f_name in file_names:
        hyp_lists[f_name] = []
        with open(f"{dir_path}/{f_name}", "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        for sample in data:
            if len(hyp_lists) == 1:
                ref_list.append(sample["gold_response"])
            if "ARDM_generation" in sample.keys():
                hyp_lists[f_name].append(sample["ARDM_generation"])
            else:
                hyp_lists[f_name].append(sample["ProAware_generation"])
            if len(hyp_lists[f_name][-1]) == 0:
                hyp_lists[f_name][-1] = "[Empty]"
    return ref_list, hyp_lists


def get_LLM_data(f_name):
    with open(f_name, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    ref_list = []
    hyp_lists = {"GPT-3.5": [], "GPT-3.5+CoT": [], "MixInit": []}
    for sample in data:
        ref_list.append(sample["gold_response"])
        hyp_lists["GPT-3.5"].append(sample["baseline_generations"]["GPT-3.5"])
        hyp_lists["GPT-3.5+CoT"].append(sample["baseline_generations"]["GPT-3.5+CoT"]["extracted_response"])
        # hyp_lists["MixInit"].append(sample["baseline_generations"]["MixInit"]["extracted_response"])
        tmp = sample["baseline_generations"]["MixInit"]["extracted_response"]
        if len(tmp)==0:
            tmp = sample["baseline_generations"]["MixInit"]["complete_output"]
            tmp = tmp[tmp.find(']')+1:].rstrip("[end]").rstrip().lstrip()
        hyp_lists["MixInit"].append(tmp)
    return ref_list, hyp_lists

def get_KEMI_data():
    ref_list = []
    hyp_lists = {"KEMI": []}
    with open("../KEMI/gen.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        sample = json.loads(line.rstrip())
        ref_list.append(sample["response"])
        hyp_lists["KEMI"].append(sample["generation"])
    return ref_list, hyp_lists

def get_multiesc_results():
    ref_list = []
    hyp_lists = {"multiesc_beam1": [], "multiesc_beam4": []}
    with open("../MultiESC/generation_results/multiesc_beam1_results.json") as f:
        data = json.loads(f.read())
    for sample in data:
        ref_list.append(sample["gold_response"])
        hyp_lists["multiesc_beam1"].append(sample["generation"])

    with open("../MultiESC/generation_results/multiesc_beam4_results.json") as f:
        data = json.loads(f.read())
    for sample in data:
        hyp_lists["multiesc_beam4"].append(sample["generation"])
    return ref_list, hyp_lists

def get_bart_vanilla_results():
    ref_list = []
    hyp_lists = {"bart_vanilla_beam1": [], "bart_vanilla_beam4": []}
    with open("../BART_vanilla/final_beam1.json") as f:
        data = json.loads(f.read())
    for sample in data:
        ref_list.append(sample["gold_response"])
        hyp_lists["bart_vanilla_beam1"].append(sample["generation"])

    with open("../BART_vanilla/final_beam4.json") as f:
        data = json.loads(f.read())
    for sample in data:
        hyp_lists["bart_vanilla_beam4"].append(sample["generation"])
    return ref_list, hyp_lists

def get_ours_finetuned_results():
    ref_list = []
    hyp_lists = {"tmp2": []}#, "multiesc_beam4": []}
    # with open("../ours_finetuned/tmp1.json") as f:
    #     data = json.loads(f.read())
    # for sample in data:
    #     ref_list.append(sample["gold_response"])
    #     hyp_lists["tmp1"].append(sample["generation"])

    with open("../ours_finetuned/tmp2.json") as f:
        data = json.loads(f.read())
    for sample in data:
        ref_list.append(sample["gold_response"])
        hyp_lists["tmp2"].append(sample["generation"])
    return ref_list, hyp_lists

def calculate_metrics(ref_list, hyp_list):
    for i, ref in enumerate(hyp_list):
        ref_list[i] = ' '.join(nltk.word_tokenize(ref_list[i].lower()))
        hyp_list[i] = ' '.join(nltk.word_tokenize(hyp_list[i].lower()))
    metric = NLGEval()
    metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
    for k in range(1,5):
        metric_res[f'Distinct-{k}'] = calc_distinct_k(hyp_list, k)
    return metric_res

def get_ours_LLM():
    ref_list = []
    hyp_lists = {"Cooper_LLM": []}#, "multiesc_beam4": []}
    # with open("../ours_finetuned/tmp1.json") as f:
    #     data = json.loads(f.read())
    # for sample in data:
    #     ref_list.append(sample["gold_response"])
    #     hyp_lists["tmp1"].append(sample["generation"])

    with open("../Cooper_LLM_results_300.json") as f:
        data = json.loads(f.read())
    for sample in data:
        ref_list.append(sample["gold_response"])
        generation = sample["generation"]
        if generation.startswith("Supporter: "):
            generation = generation[len("Supporter: "):]
        hyp_lists["Cooper_LLM"].append(generation)
    return ref_list, hyp_lists


if __name__ == "__main__":
    ref_list, hyp_lists = get_ours_LLM()
    results = dict()
    print(ref_list[:4])
    print(hyp_lists['Cooper_LLM'][:4])
    for baseline in hyp_lists.keys():
        results[baseline] = calculate_metrics(ref_list, hyp_lists[baseline])
        print(baseline)
        print(results[baseline])

