from functools import wraps


def fix_get_bboxes_output():
    """
    Because in SingleStageDetector.onnx_export, after calling
    self.bbox_head.get_bboxes, only two output values are expected: det_bboxes,
    det_labels.
    """

    def crop_output(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            output_list = function(*args, **kwargs)
            return output_list[0]

        return wrapper

    from mmdet.models.dense_heads.fovea_head import FoveaHead
    from mmdet.models.dense_heads.atss_head import ATSSHead
    heads = [FoveaHead, ATSSHead]
    for head in heads:
        head.get_bboxes = crop_output(head.get_bboxes)


def apply_all_fixes():
    fix_get_bboxes_output()
