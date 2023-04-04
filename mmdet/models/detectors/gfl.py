# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class GFL(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GFL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
    def forward_dummy(self, img):
        # type: (Tensor) -> tuple[Tensor]
        """Used for exporting TorchScript."""
        x = self.extract_feat(img)
        #cls_scores, bbox_preds = self.bbox_head(x)
        #outs = tuple(cls_scores + bbox_preds)
        outs, tower_conv = self.bbox_head(x) 
        outs = outs + tower_conv
        #outs = tuple(outs + cls_feat_list) 
        return outs