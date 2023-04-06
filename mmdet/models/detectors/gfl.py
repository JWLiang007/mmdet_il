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
                 pretrained=None):
        super(GFL, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)
    
    def forward_dummy(self, img):
        # type: (Tensor) -> tuple[Tensor]
        """Used for exporting TorchScript."""
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # cls_scores, bbox_preds = self.bbox_head(x)
        # outs = tuple(cls_scores + bbox_preds)
        return outs
