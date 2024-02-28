import torch
# from transformers.models.bert.modeling_bert import _TOKENIZER_FOR_DOC,_CHECKPOINT_FOR_DOC,_CONFIG_FOR_DOC
from transformers.models.bert.modeling_bert import *
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
# MODE_TYPE = 0

class SimpleGatLayer(nn.Module):
    def __init__(self, hdim):
        super(SimpleGatLayer, self).__init__()
        self.hdim = hdim
        self.proj_w = nn.Parameter(torch.empty(size=(hdim, hdim)))
        nn.init.xavier_uniform_(self.proj_w.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU()

    def forward(self, sentence_vec, node_embedding):
        sentence_vec = sentence_vec.matmul(self.proj_w) # bs * hdim
        node_embedding = node_embedding.matmul(self.proj_w) # bs * node_num * hdim
        attention_score = sentence_vec.unsqueeze(1).matmul(node_embedding.transpose(1,2)) # bs * node_num
        attention_score = F.softmax(attention_score, dim=-1)
        post_vec = attention_score.matmul(node_embedding).squeeze(1)
        return post_vec
        # setence_vec : bs * hdim, node_embedding: node_num * hdim

# 分别使用三个gat 聚合 各个agent的模型， 最后使用一个gat，聚合pool ouput和三个agent的状态。
class BertForTopicRanking_four_gat(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        cls_hidden_size = 2 * config.hidden_size
        self.classifier = nn.Linear(cls_hidden_size, 1)
        self.point_cls = nn.Linear(cls_hidden_size, 13)
        self.margin = 0.2
        self.init_weights()
        # state_name = ['exploration_state','comfort_state','action_state']
        self.cluster_embedding =[
            torch.Tensor(np.load('data/esconv/exploration_state_centers.npy')),
            torch.Tensor(np.load('data/esconv/comfort_state_centers.npy')),
            torch.Tensor(np.load('data/esconv/action_state_centers.npy'))
            ]
        self.gat_layer = nn.ModuleList([SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size)])
        self.gat_layer2 = SimpleGatLayer(config.hidden_size)
        self.merged_linear = nn.Linear(3 * config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cluster_index=None,
        progression_state=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        bs = labels.size()[0]
        progression_merged_state = []

        for i in range(3):
            this_progression_state = progression_state[:,i,:]
            tmp_cluster_embedding = self.cluster_embedding[i].unsqueeze(0).expand(bs,-1,-1).to(cluster_index.device) # bs * cluster_num * hdim
            topk_cluster_index = cluster_index[:,i,:].unsqueeze(-1).expand(-1,-1,tmp_cluster_embedding.size(-1)) # bs * topk_num * 1
            gatherd_embedding = torch.gather(tmp_cluster_embedding, 1 , topk_cluster_index)
            one_progression = self.gat_layer[i](this_progression_state, gatherd_embedding) # bs * hdim
            progression_merged_state.append(one_progression)

        con_merged1 = torch.cat([x.unsqueeze(1) for x in progression_merged_state], 1)
        # 可以把con-merged当成终点， 然后用当前pool的向量，与con-merged的距离，作为rank的magin。是不是有点道理的意思。
        merged_progression = self.gat_layer2(pooled_output, con_merged1)
        pooled_output_merged = torch.cat([pooled_output, merged_progression], -1)

        cls_logits = self.point_cls(pooled_output_merged)
        cross_loss = nn.CrossEntropyLoss()
        cls_loss = cross_loss(cls_logits, labels[:,0])

        logits = self.classifier(pooled_output_merged) # bs * 1

        d_id1 = labels[:,1].unsqueeze(0).repeat(bs,1)
        d_id2 = d_id1.transpose(0, 1)

        score1 = logits.squeeze(-1).unsqueeze(0).repeat(bs, 1)
        score2 = score1.transpose(0, 1)
        label1 = labels[:,0].unsqueeze(0).repeat(bs, 1)
        label2 = label1.transpose(0,1)
        pn_labels = (label1-label2) * (d_id1 == d_id2).type(torch.int)
        loss = (pn_labels > 0) * torch.relu(score2 - score1 + self.margin)
        loss = torch.sum(loss) / (torch.sum(pn_labels > 0) + 1e-6)

        loss = loss + cls_loss * 0.3

        # logits = cls_logits
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertForTopicRanking_add_progression_magin(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        cls_hidden_size = 2 * config.hidden_size
        self.classifier = nn.Linear(cls_hidden_size, 1)
        self.point_cls = nn.Linear(cls_hidden_size, 13)
        self.margin = 0.2
        self.init_weights()
        # state_name = ['exploration_state','comfort_state','action_state']
        self.cluster_embedding =[
            torch.Tensor(np.load('data/esconv/exploration_state_centers.npy')),
            torch.Tensor(np.load('data/esconv/comfort_state_centers.npy')),
            torch.Tensor(np.load('data/esconv/action_state_centers.npy'))
            ]
        self.gat_layer = nn.ModuleList([SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size)])
        self.gat_layer2 = SimpleGatLayer(config.hidden_size)
        self.merged_linear = nn.Linear(3 * config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cluster_index=None,
        progression_state=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        bs = labels.size()[0]
        progression_merged_state = []

        for i in range(3):
            this_progression_state = progression_state[:,i,:]
            tmp_cluster_embedding = self.cluster_embedding[i].unsqueeze(0).expand(bs,-1,-1).to(cluster_index.device) # bs * cluster_num * hdim
            topk_cluster_index = cluster_index[:,i,:].unsqueeze(-1).expand(-1,-1,tmp_cluster_embedding.size(-1)) # bs * topk_num * 1
            gatherd_embedding = torch.gather(tmp_cluster_embedding, 1 , topk_cluster_index)
            one_progression = self.gat_layer[i](this_progression_state, gatherd_embedding) # bs * hdim
            progression_merged_state.append(one_progression)

        # con_merged = torch.cat(progression_merged_state, -1)
        con_merged1 = torch.cat([x.unsqueeze(1) for x in progression_merged_state], 1)
        # merged_progression = self.merged_linear(con_merged) # bs * hdim
        # 可以把con-merged当成终点， 然后用当前pool的向量，与con-merged的距离，作为rank的magin。是不是有点道理的意思。
        merged_progression = self.gat_layer2(pooled_output, con_merged1) # 终点
        each_distance = F.cosine_similarity(pooled_output, merged_progression)
        pooled_output_merged = torch.cat([pooled_output, merged_progression], -1)



        cls_logits = self.point_cls(pooled_output_merged)
        cross_loss = nn.CrossEntropyLoss()
        cls_loss = cross_loss(cls_logits, labels[:,0])

        logits = self.classifier(pooled_output_merged) # bs * 1

        d_id1 = labels[:,1].unsqueeze(0).repeat(bs,1)
        d_id2 = d_id1.transpose(0, 1)

        dis1 = each_distance.unsqueeze(0).repeat(bs, 1)
        dis2 = dis1.transpose(0, 1)
        score1 = logits.squeeze(-1).unsqueeze(0).repeat(bs, 1)
        score2 = score1.transpose(0, 1)
        label1 = labels[:,0].unsqueeze(0).repeat(bs, 1)
        label2 = label1.transpose(0,1)
        pn_labels = (label1-label2) * (d_id1 == d_id2).type(torch.int)
        margin_diff = torch.clip(dis1-dis2, min=0.2, max=0.5)

        loss = (pn_labels > 0) * torch.relu(score2 - score1 + margin_diff)
        loss = torch.sum(loss) / (torch.sum(pn_labels > 0) + 1e-6)

        loss = loss + cls_loss * 0.2

        # logits = cls_logits
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertForTopicRanking_add_progression_magin_in_point_loss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        cls_hidden_size = 2 * config.hidden_size
        self.classifier = nn.Linear(cls_hidden_size, 1)
        self.point_cls = nn.Linear(cls_hidden_size, 13)
        self.margin = 0.2
        self.init_weights()
        # state_name = ['exploration_state','comfort_state','action_state']
        self.cluster_embedding =[
            torch.Tensor(np.load('data/esconv/exploration_state_centers.npy')),
            torch.Tensor(np.load('data/esconv/comfort_state_centers.npy')),
            torch.Tensor(np.load('data/esconv/action_state_centers.npy'))
            ]
        self.gat_layer = nn.ModuleList([SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size)])
        self.gat_layer2 = SimpleGatLayer(config.hidden_size)
        self.merged_linear = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.distance_linear = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cluster_index=None,
        progression_state=None,
        sub_task=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        bs = labels.size()[0]
        progression_merged_state = []

        for i in range(3):
            this_progression_state = progression_state[:,i,:]
            tmp_cluster_embedding = self.cluster_embedding[i].unsqueeze(0).expand(bs,-1,-1).to(cluster_index.device) # bs * cluster_num * hdim
            topk_cluster_index = cluster_index[:,i,:].unsqueeze(-1).expand(-1,-1,tmp_cluster_embedding.size(-1)) # bs * topk_num * 1
            gatherd_embedding = torch.gather(tmp_cluster_embedding, 1 , topk_cluster_index)
            one_progression = self.gat_layer[i](this_progression_state, gatherd_embedding) # bs * hdim
            progression_merged_state.append(one_progression)

        # con_merged = torch.cat(progression_merged_state, -1)
        con_merged1 = torch.cat([x.unsqueeze(1) for x in progression_merged_state], 1)
        # merged_progression = self.merged_linear(con_merged) # bs * hdim
        # 可以把con-merged当成终点， 然后用当前pool的向量，与con-merged的距离，作为rank的magin。是不是有点道理的意思。
        merged_progression = self.gat_layer2(pooled_output, con_merged1) # 终点
        each_distance = torch.sigmoid(self.distance_linear(merged_progression - pooled_output))
        # each_distance = pooled_output.matmul(con_merged1.transpose(1,2))
        # each_distance = F.cosine_similarity(pooled_output, merged_progression)
        # squre_distance = F.
        pooled_output_merged = torch.cat([pooled_output, merged_progression], -1)

        cls_logits = self.point_cls(pooled_output_merged)
        cross_loss = nn.CrossEntropyLoss()
        cls_loss = cross_loss(cls_logits, labels[:,0])

        logits = self.classifier(pooled_output_merged) # bs * 1

        d_id1 = labels[:,1].unsqueeze(0).repeat(bs,1)
        d_id2 = d_id1.transpose(0, 1)
        # print('margin_diff', each_distance.shape)
        dis1 = each_distance
        dis2 = dis1.transpose(0, 1)
        score1 = logits.squeeze(-1).unsqueeze(0).repeat(bs, 1)
        score2 = score1.transpose(0, 1)
        label1 = labels[:,0].unsqueeze(0).repeat(bs, 1)
        label2 = label1.transpose(0,1)
        pn_labels = (label1-label2) * (d_id1 == d_id2).type(torch.int)
        margin_diff = torch.clip(dis2-dis1, min=0.2, max=0.5)
        # if bs < 16:
        #     print("margin_diff:",margin_diff)
        # ddd = dis1 - dis2
        # print('margin diff:', margin_diff[:,0])
        loss = (pn_labels > 0) * torch.relu(score2 - score1 + margin_diff)
        loss = torch.sum(loss) / (torch.sum(pn_labels > 0) + 1e-6)

        loss = loss + cls_loss * torch.mean(margin_diff)

        # logits = cls_logits
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertForTopicRanking_add_progression_magin_only_in_point_loss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        cls_hidden_size = 2 * config.hidden_size
        self.classifier = nn.Linear(cls_hidden_size, 1)
        self.point_cls = nn.Linear(cls_hidden_size, 13)
        self.margin = 0.2
        self.init_weights()
        # state_name = ['exploration_state','comfort_state','action_state']
        self.cluster_embedding =[
            torch.Tensor(np.load('data/esconv/exploration_state_centers.npy')),
            torch.Tensor(np.load('data/esconv/comfort_state_centers.npy')),
            torch.Tensor(np.load('data/esconv/action_state_centers.npy'))
            ]
        self.gat_layer = nn.ModuleList([SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size)])
        self.gat_layer2 = SimpleGatLayer(config.hidden_size)
        self.merged_linear = nn.Linear(3 * config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cluster_index=None,
        progression_state=None,
        sub_task=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        bs = labels.size()[0]
        progression_merged_state = []

        for i in range(3):
            this_progression_state = progression_state[:,i,:]
            tmp_cluster_embedding = self.cluster_embedding[i].unsqueeze(0).expand(bs,-1,-1).to(cluster_index.device) # bs * cluster_num * hdim
            topk_cluster_index = cluster_index[:,i,:].unsqueeze(-1).expand(-1,-1,tmp_cluster_embedding.size(-1)) # bs * topk_num * 1
            gatherd_embedding = torch.gather(tmp_cluster_embedding, 1 , topk_cluster_index)
            one_progression = self.gat_layer[i](this_progression_state, gatherd_embedding) # bs * hdim
            progression_merged_state.append(one_progression)

        # con_merged = torch.cat(progression_merged_state, -1)
        con_merged1 = torch.cat([x.unsqueeze(1) for x in progression_merged_state], 1)
        # merged_progression = self.merged_linear(con_merged) # bs * hdim
        # 可以把con-merged当成终点， 然后用当前pool的向量，与con-merged的距离，作为rank的magin。是不是有点道理的意思。
        merged_progression = self.gat_layer2(pooled_output, con_merged1) # 终点
        each_distance = F.cosine_similarity(pooled_output, merged_progression)
        pooled_output_merged = torch.cat([pooled_output, merged_progression], -1)

        cls_logits = self.point_cls(pooled_output_merged)
        cross_loss = nn.CrossEntropyLoss()
        cls_loss = cross_loss(cls_logits, labels[:,0])

        logits = self.classifier(pooled_output_merged) # bs * 1

        d_id1 = labels[:,1].unsqueeze(0).repeat(bs,1)
        d_id2 = d_id1.transpose(0, 1)

        dis1 = each_distance.unsqueeze(0).repeat(bs, 1)
        dis2 = dis1.transpose(0, 1)
        score1 = logits.squeeze(-1).unsqueeze(0).repeat(bs, 1)
        score2 = score1.transpose(0, 1)
        label1 = labels[:,0].unsqueeze(0).repeat(bs, 1)
        label2 = label1.transpose(0,1)
        pn_labels = (label1-label2) * (d_id1 == d_id2).type(torch.int)
        margin_diff = torch.clip(dis1-dis2, min=0.1, max=0.5)

        loss = (pn_labels > 0) * torch.relu(score2 - score1 + self.margin)
        loss = torch.sum(loss) / (torch.sum(pn_labels > 0) + 1e-6)

        loss = loss + cls_loss * torch.mean(margin_diff)

        # logits = cls_logits
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# baseline
class BertForTopicRankingWoProgression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.margin = 0.2
        self.point_cls = nn.Linear(config.hidden_size, 13)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cluster_index=None,
        progression_state=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        bs = labels.size()[0]

        cls_logits = self.point_cls(pooled_output)
        cross_loss = nn.CrossEntropyLoss()
        cls_loss = cross_loss(cls_logits, labels[:,0])

        logits = self.classifier(pooled_output)

        d_id1 = labels[:,1].unsqueeze(0).repeat(bs,1)
        d_id2 = d_id1.transpose(0, 1)

        score1 = logits.squeeze(-1).unsqueeze(0).repeat(bs, 1)
        score2 = score1.transpose(0, 1)
        label1 = labels[:,0].unsqueeze(0).repeat(bs, 1)
        label2 = label1.transpose(0,1)
        pn_labels = (label1-label2) * (d_id1 == d_id2).type(torch.int)
        loss = (pn_labels > 0) * torch.relu(score2 - score1 + self.margin)
        loss = torch.sum(loss) / (torch.sum(pn_labels > 0) + 1e-6)
        loss += 0.1 * cls_loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class BertForTopicRankingWoProgressionLessMagin(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.margin = 0.3
        self.point_cls = nn.Linear(config.hidden_size, 13)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cluster_index=None,
        progression_state=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        bs = labels.size()[0]

        cls_logits = self.point_cls(pooled_output)
        cross_loss = nn.CrossEntropyLoss()
        cls_loss = cross_loss(cls_logits, labels[:,0])

        logits = self.classifier(pooled_output)

        d_id1 = labels[:,1].unsqueeze(0).repeat(bs,1)
        d_id2 = d_id1.transpose(0, 1)

        score1 = logits.squeeze(-1).unsqueeze(0).repeat(bs, 1)
        score2 = score1.transpose(0, 1)
        label1 = labels[:,0].unsqueeze(0).repeat(bs, 1)
        label2 = label1.transpose(0,1)
        pn_labels = (label1-label2) * (d_id1 == d_id2).type(torch.int)
        loss = (pn_labels > 0) * torch.relu(score2 - score1 + self.margin)
        loss = torch.sum(loss) / (torch.sum(pn_labels > 0) + 1e-6)
        loss += 0.2 * cls_loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class BertForTopicRankingWoProgressionmorecls(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.margin = 0.3
        self.point_cls = nn.Linear(config.hidden_size, 13)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cluster_index=None,
        progression_state=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        bs = labels.size()[0]

        cls_logits = self.point_cls(pooled_output)
        cross_loss = nn.CrossEntropyLoss()
        cls_loss = cross_loss(cls_logits, labels[:,0])

        logits = self.classifier(pooled_output)

        d_id1 = labels[:,1].unsqueeze(0).repeat(bs,1)
        d_id2 = d_id1.transpose(0, 1)

        score1 = logits.squeeze(-1).unsqueeze(0).repeat(bs, 1)
        score2 = score1.transpose(0, 1)
        label1 = labels[:,0].unsqueeze(0).repeat(bs, 1)
        label2 = label1.transpose(0,1)
        pn_labels = (label1-label2) * (d_id1 == d_id2).type(torch.int)
        loss = (pn_labels > 0) * torch.relu(score2 - score1 + self.margin)
        loss = torch.sum(loss) / (torch.sum(pn_labels > 0) + 1e-6)
        loss += cls_loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForTopicRanking_add_progression_magin_only_in_point_loss_p4g(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        cls_hidden_size = 2 * config.hidden_size
        self.classifier = nn.Linear(cls_hidden_size, 1)
        self.point_cls = nn.Linear(cls_hidden_size, 13)
        self.margin = 0.2
        self.init_weights()
        # state_name = ['exploration_state','comfort_state','action_state']
        self.cluster_embedding =[
            torch.Tensor(np.load('./p4g_data_complete_context/inquiry_state_centers.npy')),
            torch.Tensor(np.load('./p4g_data_complete_context/appeal_state_centers.npy')),
            torch.Tensor(np.load('./p4g_data_complete_context/proposition_state_centers.npy'))
            ]
        self.gat_layer = nn.ModuleList([SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size),SimpleGatLayer(config.hidden_size)])
        self.gat_layer2 = SimpleGatLayer(config.hidden_size)
        self.merged_linear = nn.Linear(3 * config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cluster_index=None,
        progression_state=None,
        sub_task=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        bs = labels.size()[0]
        progression_merged_state = []

        for i in range(3):
            this_progression_state = progression_state[:,i,:]
            tmp_cluster_embedding = self.cluster_embedding[i].unsqueeze(0).expand(bs,-1,-1).to(cluster_index.device) # bs * cluster_num * hdim
            topk_cluster_index = cluster_index[:,i,:].unsqueeze(-1).expand(-1,-1,tmp_cluster_embedding.size(-1)) # bs * topk_num * 1
            gatherd_embedding = torch.gather(tmp_cluster_embedding, 1 , topk_cluster_index)
            one_progression = self.gat_layer[i](this_progression_state, gatherd_embedding) # bs * hdim
            progression_merged_state.append(one_progression)

        # con_merged = torch.cat(progression_merged_state, -1)
        con_merged1 = torch.cat([x.unsqueeze(1) for x in progression_merged_state], 1)
        # merged_progression = self.merged_linear(con_merged) # bs * hdim
        # 可以把con-merged当成终点， 然后用当前pool的向量，与con-merged的距离，作为rank的magin。是不是有点道理的意思。
        merged_progression = self.gat_layer2(pooled_output, con_merged1) # 终点
        each_distance = F.cosine_similarity(pooled_output, merged_progression)
        pooled_output_merged = torch.cat([pooled_output, merged_progression], -1)

        cls_logits = self.point_cls(pooled_output_merged)
        cross_loss = nn.CrossEntropyLoss()
        cls_loss = cross_loss(cls_logits, labels[:,0])

        logits = self.classifier(pooled_output_merged) # bs * 1

        d_id1 = labels[:,1].unsqueeze(0).repeat(bs,1)
        d_id2 = d_id1.transpose(0, 1)

        dis1 = each_distance.unsqueeze(0).repeat(bs, 1)
        dis2 = dis1.transpose(0, 1)
        score1 = logits.squeeze(-1).unsqueeze(0).repeat(bs, 1)
        score2 = score1.transpose(0, 1)
        label1 = labels[:,0].unsqueeze(0).repeat(bs, 1)
        label2 = label1.transpose(0,1)
        pn_labels = (label1-label2) * (d_id1 == d_id2).type(torch.int)
        margin_diff = torch.clip(dis1-dis2, min=0.1, max=0.5)

        loss = (pn_labels > 0) * torch.relu(score2 - score1 + self.margin)
        loss = torch.sum(loss) / (torch.sum(pn_labels > 0) + 1e-6)

        loss = loss + cls_loss * torch.mean(margin_diff)

        # logits = cls_logits
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



MODEL_LIST={
    "0":BertForTopicRankingWoProgression,
    "1":BertForTopicRanking_four_gat,
    "2":BertForTopicRanking_add_progression_magin,
    "20":BertForTopicRanking_add_progression_magin_in_point_loss,
    "00":BertForTopicRankingWoProgressionLessMagin,
    "01":BertForTopicRankingWoProgressionmorecls,
    "21": BertForTopicRanking_add_progression_magin_only_in_point_loss,
    "40": BertForTopicRanking_add_progression_magin_only_in_point_loss_p4g,
}