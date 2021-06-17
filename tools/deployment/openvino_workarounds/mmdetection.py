import importlib
from functools import partial, wraps

import torch


def update_default_args_value(func, **updated_args):
    new_func = partial(func, **updated_args)
    return new_func


def fix_get_bboxes_output():
    """Because in SingleStageDetector.onnx_export, after calling
    self.bbox_head.get_bboxes, only two output values are expected: det_bboxes,
    det_labels."""

    def crop_output(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            output_list = function(*args, **kwargs)
            return output_list[0]

        return wrapper

    dense_heads = importlib.import_module('mmdet.models.dense_heads')
    heads = ['FoveaHead', 'ATSSHead']
    for head_name in heads:
        head_class = getattr(dense_heads, head_name)
        head_class.get_bboxes = crop_output(head_class.get_bboxes)


def fix_img_shape_type():
    """Some models (ATSS, FoveaBox) use img_metas[0]['img_shape'], which type
    is 'list'.

    To export with dynamic inputs, the type should be changed to 'tensor'.
    Can be fixed here: https://github.com/open-mmlab/mmdetection/pull/5251.
    Or you can remove support for img_metas[0]['img_shape_for_onnx'] and
    always convert values to a tensor with input dimension.
    """

    def rewrite_img_shape_in_onnx_export(function):

        @wraps(function)
        def wrapper(self, img, img_metas):
            img_metas[0]['img_shape'] = torch._shape_as_tensor(img)[2:]
            return function(self, img, img_metas)

        return wrapper

    from mmdet.models.detectors.single_stage import SingleStageDetector
    SingleStageDetector.onnx_export = rewrite_img_shape_in_onnx_export(
        SingleStageDetector.onnx_export)


def apply_all_fixes():
    fix_get_bboxes_output()
    fix_img_shape_type()
