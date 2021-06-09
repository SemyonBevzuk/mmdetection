import importlib
from functools import partial, wraps


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


def apply_all_fixes():
    fix_get_bboxes_output()
