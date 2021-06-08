def fix_foveabox_get_bboxes_output():
    """Because in SingleStageDetector.onnx_export, after calling
    self.bbox_head.get_bboxes, only two output values are expected: det_bboxes,
    det_labels."""
    from mmdet.models.dense_heads.fovea_head import FoveaHead
    original_get_bboxes = FoveaHead.get_bboxes

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        result_list = original_get_bboxes(self, cls_scores, bbox_preds,
                                          img_metas, cfg, rescale)
        return result_list[0]

    FoveaHead.get_bboxes = get_bboxes


def apply_all_fixes():
    fix_foveabox_get_bboxes_output()
