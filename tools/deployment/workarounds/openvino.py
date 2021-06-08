from functools import wraps

import onnx
import torch


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
    """
    [MO] Incorrect TopK transformation.

    TopK after MO returns indices with type int32, which can be added to
    tensors with type int64.
    """

    def topk_inds_to_int64(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            topk_values, topk_inds = function(*args, **kwargs)
            topk_inds = topk_inds.to(dtype=torch.int64)
            return topk_values, topk_inds

        return wrapper

    torch.Tensor.topk = topk_inds_to_int64(torch.Tensor.topk)


def fix_foveabox_problem():
    """
    [MO] Incorrect constant in the IR (foveabox)

    MO saves incorrect constants in the graphics,
    which causes an error when reshaping the model in IE.
    """
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


def fix_yolov3_problem():
    """
    [MO] Incorrect constant in the IR (yolov3)

    MO saves incorrect constants in the graphics,
    which causes an error when reshaping the model in IE.
    """

    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        # Get outputs x, y
        # Error when reshaping in OpenVINO
        # x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
        # y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
        # Fix
        x_center_pred = pred_bboxes[..., 0] * stride + \
            (stride * (-0.5) + x_center)
        y_center_pred = pred_bboxes[..., 1] * stride + \
            (stride * (-0.5) + y_center)
        w_pred = torch.exp(pred_bboxes[..., 2]) * w
        h_pred = torch.exp(pred_bboxes[..., 3]) * h

        decoded_bboxes = torch.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            dim=-1)
        return decoded_bboxes

    from mmdet.core.bbox.coder.yolo_bbox_coder import YOLOBBoxCoder
    YOLOBBoxCoder.decode = decode


def apply_all_fixes():
    fix_topk_inds_output_type_problem()
    fix_foveabox_problem()
    fix_yolov3_problem()
