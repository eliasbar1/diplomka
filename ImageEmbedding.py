#!/usr/bin/python
# -*- coding: utf-8 -*-

# tento kód je založený na Detectron2 Beginner's Tutorial na 
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

import os
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import cv2
import detectron2.data.transforms as T
import torch
from detectron2.structures.image_list import ImageList
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
import numpy as np
from detectron2.layers import nms
from torch import nn

class ImageEmbedding():
    def __init__(self):        
        self.cfg_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        self.cfg = self.load_config_and_model_weights()
        self.model = self.get_model()                        
        
    def get_image_embedding(self, img_list, batch_size):
        img_bgr_list=[]
        for i in img_list:
            img_bgr_list.append(cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
        
        images, batched_inputs = self.prepare_image_inputs(img_bgr_list)
        
        features = self.get_features(images)
        proposals = self.get_proposals(images, features)
        box_features, features_list = self.get_box_features(features, proposals, batch_size)
        pred_class_logits, pred_proposal_deltas = self.get_prediction_logits(features_list, proposals)
        boxes, scores, image_shapes = self.get_box_scores(pred_class_logits, pred_proposal_deltas, proposals)
        
        output_boxes = [self.get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
        
        temp = [self.select_boxes(output_boxes[i], scores[i]) for i in range(len(scores))]
        keep_boxes, max_conf = [],[]
        for keep_box, mx_conf in temp:
            keep_boxes.append(keep_box)
            max_conf.append(mx_conf)
            
        
        MIN_BOXES=1
        MAX_BOXES=10
        keep_boxes = [self.filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]

        visual_embeds = [self.get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]
        
        return visual_embeds
    
    def load_config_and_model_weights(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.cfg_path))

        # ROI HEADS SCORE THRESHOLD
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        cfg['MODEL']['DEVICE']='cpu'

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.cfg_path)
        return cfg
    
    def get_model(self):
        # build model
        model = build_model(self.cfg)

        # load weights
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        # eval mode
        model.eval()
        return model
    
    def prepare_image_inputs(self, img_list):
        cfg=self.cfg
        # Resizing the image according to the configuration
        transform_gen = T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
                )
        img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

        # Convert to C,H,W format
        convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

        batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

        # Normalizing the image
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        images = [normalizer(x["image"]) for x in batched_inputs]

        # Convert to ImageList
        images =  ImageList.from_tensors(images, self.model.backbone.size_divisibility)

        return images, batched_inputs
    
    def get_features(self, images):
        features = self.model.backbone(images.tensor)
        return features

    def get_proposals(self, images, features):
        proposals, _ = self.model.proposal_generator(images, features)
        return proposals

    def get_box_features(self, features, proposals, batch_size):
        features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
        box_features = self.model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.model.roi_heads.box_head.flatten(box_features)
        box_features = self.model.roi_heads.box_head.fc1(box_features)
        box_features = self.model.roi_heads.box_head.fc_relu1(box_features)
        box_features = self.model.roi_heads.box_head.fc2(box_features)
        
        padd=torch.zeros(1000*batch_size-box_features.shape[0], 1024)
        box_features = torch.cat((box_features, padd), 0)
        box_features = box_features.reshape(batch_size, 1000, 1024) # depends on config
        return box_features, features_list
    
    def get_prediction_logits(self, features_list, proposals):
        cls_features = self.model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        cls_features = self.model.roi_heads.box_head(cls_features)
        pred_class_logits, pred_proposal_deltas = self.model.roi_heads.box_predictor(cls_features)
        return pred_class_logits, pred_proposal_deltas

    def get_box_scores(self, pred_class_logits, pred_proposal_deltas, proposals):
        box2box_transform = Box2BoxTransform(weights=self.cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        smooth_l1_beta = self.cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        outputs = FastRCNNOutputs(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
        )

        boxes = outputs.predict_boxes()
        scores = outputs.predict_probs()
        image_shapes = outputs.image_shapes

        return boxes, scores, image_shapes
    
    def get_output_boxes(self, boxes, batched_inputs, image_size):
        proposal_boxes = boxes.reshape(-1, 4).clone()
        scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
        output_boxes = Boxes(proposal_boxes)
        return output_boxes
    
    def select_boxes(self, output_boxes, scores):
        test_score_thresh = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        test_nms_thresh = self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        cls_prob = scores.detach()

        out_boxes_to_padd=output_boxes.tensor.detach()
        padd=torch.zeros(80000-out_boxes_to_padd.shape[0], 4)
        out_boxes_padded = torch.cat((out_boxes_to_padd, padd), 0)
        
        cls_boxes = out_boxes_padded.reshape(1000,80,4)
        max_conf = torch.zeros((cls_boxes.shape[0]))
        for cls_ind in range(0, cls_prob.shape[1]-1):
            cls_scores = cls_prob[:, cls_ind+1]
            padd=torch.zeros(1000-len(cls_scores))
            cls_scores=torch.cat((cls_scores, padd), 0)            
            det_boxes = cls_boxes[:,cls_ind,:]
            keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
        keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
        return keep_boxes, max_conf

    def filter_boxes(self, keep_boxes, max_conf, min_boxes, max_boxes):
        if len(keep_boxes) < min_boxes:
            keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
        elif len(keep_boxes) > max_boxes:
            keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
        return keep_boxes
    
    def get_visual_embeds(self, box_features, keep_boxes):
        return box_features[keep_boxes.copy()]