import os

from base_test_case import PatcheTestCaseBase


class PatcheTestCase(PatcheTestCaseBase):

    def test_fix_yolov3_problem_is_actual(self):
        model_name = 'yolo/yolov3_d53_mstrain-608_273e_coco'
        config_path = '/tmp/openmmlab/yolo/' \
            'yolov3_d53_mstrain-608_273e_coco/config.py'
        snapshot_path = '/tmp/openmmlab/snapshots/' \
            'yolov3_d53_mstrain-608_273e_coco-139f5633.pth'
        onnx_model_path = os.path.join(self.model_folder, 'config_yolov3.onnx')
        openvino_model_path = os.path.join(self.model_folder,
                                           'config_yolov3.xml')
        pytorch2openvino_args = [
            config_path, snapshot_path, '--output-file', onnx_model_path,
            '--opset-version', '11', '--dynamic-export',
            '--not_strip_doc_string'
        ]
        pytorch2openvino_args_skip = pytorch2openvino_args + [
            '--skip_fixes', 'fix_yolov3_problem'
        ]

        from pytorch2openvino import main as pytorch2openvino_export
        pytorch2openvino_export(pytorch2openvino_args_skip)
        from test_models import check_model_with_imgs
        try:
            check_model_with_imgs(
                model_name,
                config_path,
                openvino_model_path,
                metrics=('bbox', ))
        except AssertionError as e:
            print(str(e))
        else:
            raise RuntimeError(
                'The export worked successfully before applying fixes.')

        pytorch2openvino_export(pytorch2openvino_args)
        try:
            check_model_with_imgs(
                model_name,
                config_path,
                openvino_model_path,
                metrics=('bbox', ))
        except RuntimeError as e:
            print(str(e))
