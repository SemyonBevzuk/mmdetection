import torch
import onnx
from functools import wraps


def rename_input_onnx(onnx_model_path, old_name, new_name):
    onnx_model = onnx.load(onnx_model_path)
    for node in onnx_model.graph.node:
        for i in range(len(node.input)):
            if node.input[i] == old_name:
                node.input[i] = new_name

    for input in onnx_model.graph.input:
        if input.name == old_name:
            input.name = new_name

    onnx.save(onnx_model, onnx_model_path)


def fix_topk_inds_output_type_problem():
    '''
    [MO] Incorrect TopK transformation.
    TopK after MO returns indices with type int32,
    which can be added to tensors with type int64.
    '''
    def topk_inds_to_int64(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            topk_values, topk_inds = function(*args, **kwargs)
            topk_inds = topk_inds.to(dtype=torch.int64)
            return topk_values, topk_inds

        return wrapper
    torch.Tensor.topk = topk_inds_to_int64(torch.Tensor.topk)


def fix_foveabox_problem():
    '''
    [MO] Incorrect constant in the IR (foveabox)
    MO saves incorrect constants in the graphics,
    which causes an error when reshaping the model in IE.
    '''
    from mmdet.core import multiclass_nms

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           featmap_sizes,
                           point_list,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(point_list)
        det_bboxes = []
        det_scores = []
        for cls_score, bbox_pred, featmap_size, stride, base_len, (y, x) \
                in zip(cls_scores, bbox_preds, featmap_sizes, self.strides,
                       self.base_edge_list, point_list):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).exp()
            nms_pre = cfg.get('nms_pre', -1)
            if (nms_pre > 0) and (scores.shape[0] > nms_pre):
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                y = y[topk_inds]
                x = x[topk_inds]
            x1 = (stride * x + (-base_len) * bbox_pred[:, 0]). \
                clamp(min=0, max=img_shape[1] - 1)
            y1 = (stride * y + (-base_len) * bbox_pred[:, 1]). \
                clamp(min=0, max=img_shape[0] - 1)
            x2 = (stride * x + base_len * bbox_pred[:, 2]). \
                clamp(min=0, max=img_shape[1] - 1)
            y2 = (stride * y + base_len * bbox_pred[:, 3]). \
                clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)
            det_bboxes.append(bboxes)
            det_scores.append(scores)
        det_bboxes = torch.cat(det_bboxes)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_scores = torch.cat(det_scores)
        padding = det_scores.new_zeros(det_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        det_scores = torch.cat([det_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(det_bboxes, det_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels

    from mmdet.models.dense_heads.fovea_head import FoveaHead
    FoveaHead._get_bboxes_single = _get_bboxes_single


def fix_foveabox_get_bboxes_output():
    '''
    Because in SingleStageDetector.onnx_export,
    after calling self.bbox_head.get_bboxes,
    only two output values are expected: det_bboxes, det_labels
    '''
    # from mmdet.ops.nms import NMSop
    # original_forward = NMSop.forward
    pass


def apply_all_fixes():
    fix_topk_inds_output_type_problem()
    fix_foveabox_problem()
    fix_foveabox_get_bboxes_output()
