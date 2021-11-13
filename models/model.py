# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .utils import masked_softmax, TransformerEncoder, ROIAlign
from .bmn import BoundaryMatchingNetwork
from .gtad import GTAD


class EventDetection(nn.Module):
    def __init__(self, cfg):
        super(EventDetection, self).__init__()
        self.use_env_linear = cfg.MODEL.ENV_HIDDEN_DIM is not None
        self.use_agent_linear = cfg.MODEL.AGENT_HIDDEN_DIM is not None

        if self.use_env_linear:
            self.env_linear = nn.Linear(cfg.MODEL.ENV_DIM, cfg.MODEL.ENV_HIDDEN_DIM)
        if self.use_agent_linear:
            self.agent_linear = nn.Linear(cfg.MODEL.AGENT_DIM, cfg.MODEL.AGENT_HIDDEN_DIM)

        self.agents_fuser = TransformerEncoder(cfg)
        #self.agents_environment_fuser = TransformerEncoder(cfg)

        self.bmm_name = cfg.MODEL.BOUNDARY_MATCHING_MODULE
        if self.bmm_name == 'bmn':
            self.event_detector = BoundaryMatchingNetwork(cfg)
        elif self.bmm_name == 'gtad':
            self.event_detector = GTAD(cfg)
        
        self.roi_aligner = ROIAlign(
            output_size=(7, 7),
            spatial_scale=1.0/16,
            sampling_ratio=0,
        )

        self.attention_steps = cfg.TRAIN.ATTENTION_STEPS
        self.topk_hard_attention = cfg.MODEL.TOPK_AGENTS

    def remove_agent(self, featmap, agent_boxes, agent_mask):
        agent_boxes = agent_boxes * 1.0/16
        agent_boxes[:2] = torch.floor(agent_boxes[:2])
        agent_boxes[2:] = torch.ceil(agent_boxes[2:])
        agent_boxes = agent_boxes.to(torch.int32).tolist()
        agent_mask = agent_mask.tolist()
        for box, is_selected in zip(agent_boxes, agent_mask):
            if is_selected:
                featmap[:, box[1]:box[3], box[0]:box[2]] = 0
        env_feat = torch.mean(featmap.view([featmap.shape[0], -1]), dim=-1)
        return env_feat

    def fuse_agent(self, v_featmaps, v_env_feats, v_agent_feats, v_agent_boxes):
        tmprl_sz, ft_sz = v_env_feats.shape

        selected_env_feats = []
        selected_act_feats = []
        for t in range(tmprl_sz):
            if t not in v_agent_feats:
                selected_env_feats.append(v_env_feats[t])
                selected_act_feats.append(torch.zeros_like(v_env_feats[t]))
                continue
            agent_feats = v_agent_feats[t]
            ae_feats = torch.unsqueeze(v_env_feats[t], 0) + agent_feats

            #hard_attn_masks = masks
            l2_norm = torch.norm(ae_feats, dim=-1)  # n_boxes
            l2_norm_softmax = F.softmax(l2_norm, dim=-1)  # n_boxes

            # Adaptive threshold is 1 / number of bounding boxes:
            ada_thresh = 1. / torch.full_like(l2_norm_softmax, l2_norm.shape[0])

            # Generate hard attention masks
            hard_attn_masks = l2_norm_softmax >= ada_thresh  # n_boxes
            selected_env_feats.append(self.remove_agent(v_featmaps[t], v_agent_boxes[t], hard_attn_masks))

            agent_feat = self.agents_fuser(ae_feats.unsqueeze(1), key_padding_mask=~hard_attn_masks.unsqueeze(0))  # n_boxes x keep_mask x feat_dim
            #fuser_output = fuser_input * hard_attn_masks.permute(1, 0).contiguous().unsqueeze(-1)
            act_feat = torch.sum(agent_feat, dim=0) / torch.sum(hard_attn_masks, dim=-1, keepdim=True)  # keep_mask x feat_dim
            selected_act_feats.append(act_feat.squeeze(0))

        selected_env_feats = torch.stack(selected_env_feats, dim=0)
        selected_act_feats = torch.stack(selected_act_feats, dim=0)

        return selected_env_feats, selected_act_feats

    def single_featproc(self, featmap=None, agent_boxes=None):
        agent_features = {}
        for i, boxes in agent_boxes.items():
            box_feats = self.roi_aligner(featmap[i:i+1, ...], boxes)
            agent_features[i] = torch.mean(box_feats.view(list(box_feats.shape[:-2]) + [-1]), dim=-1)
        env_features = torch.mean(featmap.view(list(featmap.shape[:2]) + [-1]), dim=-1)

        if self.use_env_linear and env_features is not None:
            env_features = self.env_linear(env_features)
        if self.use_agent_linear and agent_features is not None:
            agent_features = self.agent_linear(agent_features)

        return self.fuse_agent(featmap, env_features, agent_features, agent_boxes)


    def forward(self, featmaps=None, batch_agent_boxes=None):
        batch_env_feats = []
        batch_act_feats = []

        for featmap, agent_boxes in zip(featmaps, batch_agent_boxes):
            env_feats, act_feats = self.single_featproc(featmap, agent_boxes)
            batch_env_feats.append(env_feats)
            batch_act_feats.append(act_feats)
        
        batch_env_feats = torch.stack(batch_env_feats, dim=0)
        batch_act_feats = torch.stack(batch_act_feats, dim=0)

        env_outputs = self.event_detector(batch_env_feats.permute(0, 2, 1))
        act_outputs = self.event_detector(batch_act_feats.permute(0, 2, 1))
        return env_outputs, act_outputs
