import json
import os
import sys
import unittest
from subprocess import run, CalledProcessError, PIPE


def print_configuration_description():
    print('\tCurrent configuration:')
    python_version = '.'.join([str(i) for i in sys.version_info[0:3]])
    print(f'Python {python_version}')
    import torch
    print(f'PyTorch {torch.__version__}')
    print(f'CUDA {torch.version.cuda}')
    import mmcv
    print(f'MMCV {mmcv.__version__}')
    import onnx
    print(f'ONNX {onnx.__version__}')
    import onnxruntime
    print(f'ONNX Runtime {onnxruntime.__version__}')


class ExportModelsTestCase(unittest.TestCase):
    root_dir = os.path.join('/tmp', 'openmmlab')
    pytorch2onnx_path = os.path.join('tools', 'deployment', 'pytorch2onnx.py')

    def download_model(self, url):
        snapshots_dir = os.path.join(self.root_dir, 'snapshots')
        os.makedirs(snapshots_dir, exist_ok=True)
        path = os.path.join(snapshots_dir, os.path.basename(url))
        if not os.path.exists(path):
            run(f'wget {url} -P {snapshots_dir}', check=True, shell=True)
        return path

    def get_output_path(self, model_name):
        output_dir = os.path.join(self.root_dir, 'onnx_export')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, model_name + '.onnx')
        return path

    def check_output(self, pytorch2onnx_output):
        import re
        pattern = 'Successfully exported ONNX model:'
        if re.search(pattern, pytorch2onnx_output):
            pattern = 'The numerical values are the same between Pytorch and ONNX'
            if re.search(pattern, pytorch2onnx_output):
                print('Successfully exported. The numerical values are the same between PyTgitorch and ONNX.')
            else:
                raise AssertionError('Unsuccessfully exported. The numerical values are different between Pytorch and ONNX.')
        else:
            raise AssertionError('Unsuccessfully exported.')

    def run_pytorch2onnx_test(self, model_name, config_path, snapshot_url, opset_version):
        snapshot_path = self.download_model(snapshot_url)
        output_file = self.get_output_path(model_name)

        error = None
        try:  
            pytorch2onnx_output = run(['python', self.pytorch2onnx_path,
                    config_path, snapshot_path,
                    '--output-file', output_file,
                    '--opset-version', opset_version,
                    '--verify'],
                    capture_output=True, text=True, check=True)
            print(pytorch2onnx_output)
            pytorch2onnx_output = pytorch2onnx_output.stdout.strip("\n")
        except CalledProcessError as ex:
            error = 'Test script failure.\n' + ex.stderr
        if error is not None:
            raise RuntimeError(error)
        
        self.check_output(pytorch2onnx_output)

    def test_retinanet(self):
        model_name = 'retinanet'
        config_path = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
        snapshot_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
                    'retinanet/retinanet_r50_fpn_1x_coco/' \
                    'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        opset_version = '11'

        self.run_pytorch2onnx_test(model_name, config_path, snapshot_url, opset_version)

    def test_ssd(self):
        model_name = 'ssd'
        config_path = 'configs/ssd/ssd300_coco.py'
        snapshot_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
                    'ssd/ssd300_coco/' \
                    'ssd300_coco_20200307-a92d2092.pth'
        opset_version = '11'

        self.run_pytorch2onnx_test(model_name, config_path, snapshot_url, opset_version)

    def test_yolov3(self):
        model_name = 'yolov3'
        config_path = 'configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
        snapshot_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
                    'yolo/yolov3_d53_mstrain-608_273e_coco/' \
                    'yolov3_d53_mstrain-608_273e_coco-139f5633.pth'
        opset_version = '11'

        self.run_pytorch2onnx_test(model_name, config_path, snapshot_url, opset_version)

    def test_fsaf(self):
        model_name = 'fsaf'
        config_path = 'configs/fsaf/fsaf_r50_fpn_1x_coco.py'
        snapshot_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
                    'fsaf/fsaf_r50_fpn_1x_coco/' \
                    'fsaf_r50_fpn_1x_coco-94ccc51f.pth'
        opset_version = '11'

        self.run_pytorch2onnx_test(model_name, config_path, snapshot_url, opset_version)

    def test_faster_rcnn(self):
        model_name = 'faster_rcnn'
        config_path = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        snapshot_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
                    'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/' \
                    'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        opset_version = '11'

        self.run_pytorch2onnx_test(model_name, config_path, snapshot_url, opset_version)


if __name__ == '__main__':
    #tc = ExportModelsTestCase()
    #tc.test_faster_rcnn()
    #print_configuration_description()
    unittest.main()
