# coding=utf-8
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

import os
import torch
import warnings
import mmcv
from collections import OrderedDict
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.parallel import MMDistributedDataParallel
from mmdet.core import distance2bbox,multiclass_nms
import torch.nn.functional as F


@DETECTORS.register_module()
class GFLLwf(SingleStageDetector):
    """Incremental object detector based on GFL.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ori_config_file=None,
                 ori_checkpoint_file=None,
                 ori_num_classes=40,
                 top_k=100,
                 dist_loss_weight=1
                 ):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained)
        self.ori_checkpoint_file = ori_checkpoint_file
        self.ori_num_classes = ori_num_classes
        self.top_k = top_k
        self.dist_loss_weight = dist_loss_weight
        self.init_detector(ori_config_file, ori_checkpoint_file)

        # student_modules = dict(self.ori_model.named_modules())
        # teacher_modules = dict(self.named_modules())
        
        # self.buffer = dict()

        # def regitster_hooks(teacher_module,student_module):
        #     def hook_teacher_forward(module, input, output):
        #         self.buffer[teacher_module].append(input)

        #     def hook_student_forward(module, input, output):
        #         self.buffer[student_module].append(input)
        #     return hook_teacher_forward,hook_student_forward
        
        # module_name = 'bbox_head.loss_cls'
        # student_module = 'stu_' + module_name.replace('.',"_")
        # teacher_module = 'tea_' + module_name.replace('.',"_")
        # self.buffer[student_module]=[]
        # self.buffer[teacher_module]=[]
        # hook_teacher_forward,hook_student_forward = regitster_hooks(teacher_module ,student_module )
        # teacher_modules[module_name].register_forward_hook(hook_teacher_forward)
        # student_modules[module_name].register_forward_hook(hook_student_forward)

    def _load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=False, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                          v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        added_branch_weight = self.bbox_head.gfl_cls.weight[self.ori_num_classes:, ...]
        added_branch_bias = self.bbox_head.gfl_cls.bias[self.ori_num_classes:, ...]
        state_dict['bbox_head.gfl_cls.weight'] = torch.cat(
            (state_dict['bbox_head.gfl_cls.weight'], added_branch_weight), dim=0)
        state_dict['bbox_head.gfl_cls.bias'] = torch.cat(
            (state_dict['bbox_head.gfl_cls.bias'], added_branch_bias), dim=0)
        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict, strict, logger)
        else:
            load_state_dict(self, state_dict, strict, logger)

    def init_detector(self, config, checkpoint_file):
        """Initialize detector from config file.

        Args:
            config (str): Config file path or the config
                object.
            checkpoint_file (str): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
        assert os.path.isfile(checkpoint_file), '{} is not a valid file'.format(checkpoint_file)
        ##### init original model & frozen it #####
        # build model
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.model.bbox_head.num_classes = self.ori_num_classes
        ori_model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))#test_cfg=cfg.test_cfg
        # init weight before load checkpoints
        self.init_weights()
        ori_model.init_weights()
        # load checkpoint
        load_checkpoint(ori_model, checkpoint_file)
        # set to eval mode
        ori_model.eval()
        ori_model.forward = ori_model.forward_dummy
        # set requires_grad of all parameters to False
        for param in ori_model.parameters():
            param.requires_grad = False

        ##### init original branchs of new model #####
        self._load_checkpoint_for_new_model(checkpoint_file)

        self.ori_model = ori_model

    def model_forward(self, img, img_metas, gt_bboxes, gt_labels):
        """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            img (Tensor): Input to the model.

        Returns:
            outs (Tuple(List[Tensor])): Three model outputs.
                # cls_scores (List[Tensor]): Classification scores for each FPN level.
                # bbox_preds (List[Tensor]): BBox predictions for each FPN level.
                # centernesses (List[Tensor]): Centernesses predictions for each FPN level.
        """
        # forward the model without gradients
        with torch.no_grad():
            outs = self.ori_model(img)
            keep_inds = self.ori_model.bbox_head.get_bboxes(
                *outs, img_metas=img_metas, ret_inds = True)

        return outs, keep_inds

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels):
        # get original model outputs
        ori_outs,keep_inds = self.model_forward(img,img_metas, gt_bboxes, gt_labels)

        # get new model outputs
        x = self.extract_feat(img)
        outs = self.bbox_head(x)


        # ori_cls_scores , ori_bbox_preds = ori_outs
        # num_imgs = len(ori_cls_scores[0])
        # ori_cls_scores = [
        #     cls_score.permute(0, 2, 3, 1).reshape(
        #         num_imgs, -1, self.ori_model.bbox_head.cls_out_channels)
        #     for cls_score in ori_outs[0]
        # ]
        # ori_cls_scores = torch.cat(ori_cls_scores, dim=1)
        # ori_bbox_preds = [
        #     bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 68) #ori:4
        #     for bbox_pred in ori_outs[1]
        # ]
        # ori_bbox_preds = torch.cat(ori_bbox_preds, dim=1)
        # cls_scores = [
        #     cls_score.permute(0, 2, 3, 1).reshape(
        #         num_imgs, -1, self.bbox_head.cls_out_channels)
        #     for cls_score in outs[0]
        # ]
        # cls_scores = torch.cat(cls_scores, dim=1)
        # bbox_preds = [
        #     bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 68) #ori:4
        #     for bbox_pred in outs[1]
        # ]
        # bbox_preds = torch.cat(bbox_preds, dim=1)
        # dets = []
        # indxs = []
        # for i in range(num_imgs):
        #     det, _ ,indx= multiclass_nms(ori_bbox_preds[i], ori_cls_scores[i],   score_thr = 0.05, nms_cfg = dict(type='nms', iou_threshold=0.3),
        #                                                 max_num=-1,return_inds=True)
        #     dets.append(det)
        #     indxs.append(indx)
        # calculate losses including general losses of new model and distillation losses of original model
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas) + \
            (keep_inds,
             self.ori_num_classes, self.dist_loss_weight,  ori_outs)

        losses = self.bbox_head.loss(*loss_inputs)
        return losses
