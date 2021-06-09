from functools import wraps


def update_default_arg_value(func, arg_name, arg_value):
    defaults = func.__defaults__
    varnames = func.__code__.co_varnames
    arg_inx = varnames.index(arg_name)
    arg_inx_defaults = arg_inx - (len(varnames) - len(defaults)) + 1
    new_defaults = defaults[:arg_inx_defaults] + (
        arg_value, ) + defaults[arg_inx_defaults + 1:]
    func.__defaults__ = new_defaults


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

    from mmdet.models.dense_heads.fovea_head import FoveaHead
    from mmdet.models.dense_heads.atss_head import ATSSHead
    heads = [FoveaHead, ATSSHead]
    for head in heads:
        head.get_bboxes = crop_output(head.get_bboxes)


def apply_all_fixes():
    fix_get_bboxes_output()
