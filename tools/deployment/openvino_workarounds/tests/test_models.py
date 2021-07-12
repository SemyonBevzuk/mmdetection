import json
import os
import os.path as osp
import subprocess
import sys
import unittest
from functools import wraps
from subprocess import PIPE, CalledProcessError, run
from time import time

import colorama
import mmcv


def timer(function):

    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time()
        result = function(*args, **kwargs)
        end = time()
        print(f'Function "{function.__name__}" '
              f'took {end-start} seconds to complete.')
        return result

    return wrapper


class OpenvinoModelsTestCase(unittest.TestCase):
    root_dir = osp.join('/tmp', 'openmmlab')
    coco_dir = osp.join(root_dir, 'data', 'coco')
    snapshots_dir = osp.join(root_dir, 'snapshots')

    mmdetection_dir = os.getcwd()
    config_path_root = osp.join(mmdetection_dir, 'configs')
    config_name = 'config'

    @staticmethod
    def shorten_annotation(src_path, dst_path, num_images):
        with open(src_path) as read_file:
            content = json.load(read_file)
            selected_indexes = sorted(
                [item['id'] for item in content['images']])
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [
                item for item in content['images']
                if item['id'] in selected_indexes
            ]
            content['annotations'] = [
                item for item in content['annotations']
                if item['image_id'] in selected_indexes
            ]
            content['licenses'] = [
                item for item in content['licenses']
                if item['id'] in selected_indexes
            ]
        with open(dst_path, 'w') as write_file:
            json.dump(content, write_file)

    @classmethod
    def setUpClass(cls):
        test_on_full = False
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017.zip')):
            run(
                f'wget --no-verbose '
                f'http://images.cocodataset.org/zips/val2017.zip'
                f'-P {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017')):
            run(
                f'unzip {osp.join(cls.coco_dir, "val2017.zip")}'
                f'-d {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(
                osp.join(cls.coco_dir, 'annotations_trainval2017.zip')):
            run(
                f'wget --no-verbose http://images.cocodataset.org/'
                f'annotations/annotations_trainval2017.zip -P {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(
                osp.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            run(
                f'unzip'
                f' -o {osp.join(cls.coco_dir, "annotations_trainval2017.zip")}'
                f' -d {cls.coco_dir}',
                check=True,
                shell=True)
        if test_on_full:
            shorten_to = 5000
        else:
            shorten_to = 10
        annotation_file = osp.join(
            cls.coco_dir,
            f'annotations/instances_val2017_short_{shorten_to}.json')
        cls.shorten_annotation(
            osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
            annotation_file, shorten_to)

    def prerun(self, config_path, test_dir, cfg_options=None):
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = osp.join(test_dir, self.config_name + '.py')
        cfg = mmcv.Config.fromfile(config_path)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)

        annotation_file = osp.join(
            self.coco_dir, 'annotations/instances_val2017_short_10.json')
        update_args = {
            'data_root': f'{self.coco_dir}/',
            'data.test.ann_file': annotation_file,
            'data.test.img_prefix': osp.join(self.coco_dir, 'val2017/'),
        }
        cfg.merge_from_dict(update_args)
        with open(target_config_path, 'wt') as config_file:
            config_file.write(cfg.pretty_text)
        return target_config_path

    @staticmethod
    def collect_ap(path):
        ap = []
        beginning = \
            'Average Precision  (AP) ' \
            '@[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
        with open(path) as read_file:
            content = [line.strip() for line in read_file.readlines()]
            for line in content:
                if line.startswith(beginning):
                    ap.append(float(line.replace(beginning, '')))
        assert ap, 'could not parse metrics file'
        return ap

    def download_model(self, url):
        os.makedirs(self.snapshots_dir, exist_ok=True)
        path = osp.join(self.snapshots_dir, os.path.basename(url))
        if not os.path.exists(path):
            run(f'wget {url} -P {self.snapshots_dir}', check=True, shell=True)
        return path

    def get_output_path(self, model_name):
        output_dir = osp.join(self.root_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def check_pytorch2openvino_output(self, pytorch2onnx_output):
        import re
        pattern = 'Successfully exported ONNX model:'
        if re.search(pattern, pytorch2onnx_output):
            print('Successfully exported to ONNX.')
        else:
            raise AssertionError('Unsuccessfully exported to ONNX.')
        pattern = 'Successfully exported OpenVINO model:'
        if re.search(pattern, pytorch2onnx_output):
            print('Successfully exported to OpenVINO.')
        else:
            raise AssertionError('Unsuccessfully exported to OpenVINO.')

    def check_files(self, output_folder):
        onnx_file = osp.join(output_folder, self.config_name + '.onnx')
        if not os.path.isfile(onnx_file):
            print('The ONNX file has not been generated by python2onnx.')
            raise RuntimeError(
                'The ONNX file has not been generated by python2onnx.')
        else:
            print(
                'The ONNX file has been successfully generated by python2onnx:'
                f' {onnx_file}')

        xml_file = osp.join(output_folder, self.config_name + '.xml')
        bin_file = osp.join(output_folder, self.config_name + '.bin')
        if not os.path.isfile(xml_file) or not os.path.isfile(bin_file):
            print('OpenVino files have not been generated by mo.py.')
            raise RuntimeError(
                'OpenVino files have not been generated by mo.py.')
        else:
            print('OpenVino files have been successfully generated by mo.py: '
                  f'{xml_file}, {bin_file}')

    @timer
    def run_pytorch2openvino(self, config_path, checkpoint_path, output_file,
                             opset_version):
        error = None
        pytorch2openvino_path = os.path.join(self.mmdetection_dir, 'tools',
                                             'deployment',
                                             'pytorch2openvino.py')
        try:
            pytorch2openvino_args = f'{pytorch2openvino_path} ' \
                f'{config_path} {checkpoint_path} ' \
                f'--output-file {output_file}' \
                f' --opset-version {opset_version} ' \
                f'--dynamic-export '
            pytorch2openvino_args += '--not_strip_doc_string '

            command = f'python {pytorch2openvino_args}'
            print(f'Args for pytorch2openvino: {command}')
            process = subprocess.Popen(
                '/bin/bash',
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            pytorch2openvino_output, pytorch2openvino_errors = \
                process.communicate(command.encode())
        except CalledProcessError as ex:
            error = 'Test script failure.\n' + ex.stderr
        if error is not None:
            raise RuntimeError(error)

        return pytorch2openvino_output.decode()

    def run_export(self, model_name, config_path, checkpoint_url,
                   opset_version):
        checkpoint_path = self.download_model(checkpoint_url)
        output_folder = self.get_output_path(model_name)
        onnx_output_file = osp.join(output_folder, self.config_name + '.onnx')

        pytorch2openvino_output = self.run_pytorch2openvino(
            config_path, checkpoint_path, onnx_output_file, opset_version)

        self.check_pytorch2openvino_output(pytorch2openvino_output)
        self.check_files(output_folder)

        return output_folder

    def get_log_name(self, model_path):
        if model_path.find('.onnx'):
            return 'test_onnx.log'
        elif model_path.find('.xml'):
            return 'test_openvino.log'
        else:
            RuntimeError('The file must have extensions .onnx or .xml.')

    @timer
    def run_test_exported(self, config_path, model_path, metrics):
        log_file = osp.join(
            os.path.dirname(model_path), self.get_log_name(model_path))
        error = None
        test_exported_path = os.path.join(self.mmdetection_dir, 'tools',
                                          'deployment', 'openvino_workarounds',
                                          'tests', 'test_exported.py')
        with open(log_file, 'w') as log_f:
            try:
                run([
                    'python', test_exported_path, config_path, model_path,
                    '--out', 'res.pkl', '--eval', *metrics
                ],
                    stdout=log_f,
                    stderr=PIPE,
                    check=True)
            except CalledProcessError as ex:
                error = 'Test script failure.\n' + ex.stderr.decode(
                    sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)
        return log_file

    def check_metrics_are_close(self, current_outputs, expected_outputs,
                                metrics, threshold):
        print('Expected outputs:', expected_outputs)

        ap = self.collect_ap(current_outputs)
        with open(expected_outputs) as read_file:
            content = json.load(read_file)
        reference_ap = content['map']
        for expected, actual, m in zip(reference_ap, ap, metrics):
            print(f'Actual: {actual}, Expected: {expected}')
            if abs(actual - expected) > threshold:
                raise AssertionError(
                    f'{m}:'
                    f'{actual} (actual) - {expected} (expected) > {threshold}')

    def check_model_with_imgs(self,
                              model_name,
                              config_path,
                              model_path,
                              metrics=('bbox', )):
        log_file = self.run_test_exported(config_path, model_path, metrics)

        expected_output_file = osp.join(
            os.path.dirname(__file__), 'expected_outputs', 'public',
            f'{model_name}-10.json')
        thr = 0.02
        self.check_metrics_are_close(log_file, expected_output_file, metrics,
                                     thr)

    def run_export_test(self,
                        model_name,
                        config_path,
                        checkpoint_url,
                        opset_version,
                        cfg_options=None,
                        metrics=('bbox', )):
        print(f'\t{model_name}, opset {opset_version}')
        config_path = self.prerun(config_path,
                                  self.get_output_path(model_name),
                                  cfg_options)

        output_folder = self.run_export(model_name, config_path,
                                        checkpoint_url, opset_version)
        onnx_openmmlab_model_path = osp.join(output_folder,
                                             self.config_name + '.onnx')
        openvino_model_path = osp.join(output_folder,
                                       self.config_name + '.xml')
        models = [('ONNX', onnx_openmmlab_model_path),
                  ('OpenVINO IR', openvino_model_path)]
        for name, model_path in models:
            try:
                self.check_model_with_imgs(model_name, config_path, model_path,
                                           metrics)
                print(
                    f'{colorama.Fore.GREEN}'
                    f'For {name}, metrics on 10 imgs are the same as expected.'
                    f'{colorama.Style.RESET_ALL}')
            except AssertionError:
                print(f'{colorama.Fore.RED}'
                      f'For {name}, metrics do not match.'
                      f'{colorama.Style.RESET_ALL}')
            except RuntimeError:
                print(f'{colorama.Fore.YELLOW}'
                      'RuntimeError from test_exported.py.'
                      f'{colorama.Style.RESET_ALL}')
        print()

    def upgrade_ssd_version(self, url):
        checkpoint_path = self.download_model(url)
        checkpoint_upgrade_path = checkpoint_path.replace(
            '.pth', '_upgrade.pth')
        model_converters_path = os.path.abspath(
            osp.join(
                os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                '..', 'tools', 'model_converters'))
        upgrade_ssd_version_path = osp.join(model_converters_path,
                                            'upgrade_ssd_version.py')

        command = f'python {upgrade_ssd_version_path} ' \
                  f'{checkpoint_path} {checkpoint_upgrade_path} '
        print(f'Args for upgrade_ssd_version: {command}')
        run(command, check=True, shell=True)

        return checkpoint_upgrade_path

    @staticmethod
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
        from mo.utils.version import get_version
        print(f'OpenVINO MO {get_version()}')
        import openvino.inference_engine as ie
        print(f'OpenVINO IE {ie.__version__}')
        print()

    # Tests:
    def test_ssd(self):
        model_name = 'ssd/ssd300_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'ssd/ssd300_coco/' \
            'ssd300_coco_20200307-a92d2092.pth'
        # https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd#notice
        checkpoint_url = self.upgrade_ssd_version(checkpoint_url)

        # With this checkpoint, the metric drops from 0.256 to 0.233.
        # Checked using 'tools/deployment/test.py' on 5000 COCO imgs.
        '''
        checkpoint_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'ssd/ssd300_coco/' \
            'ssd300_coco_20210604_193052-b61137df.pth'
        '''
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_yolov3(self):
        model_name = 'yolo/yolov3_d53_mstrain-608_273e_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'yolo/yolov3_d53_mstrain-608_273e_coco/' \
            'yolov3_d53_mstrain-608_273e_coco-139f5633.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_fsaf(self):
        model_name = 'fsaf/fsaf_r50_fpn_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'fsaf/fsaf_r50_fpn_1x_coco/' \
            'fsaf_r50_fpn_1x_coco-94ccc51f.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_retinanet(self):
        model_name = 'retinanet/retinanet_r50_fpn_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'retinanet/retinanet_r50_fpn_1x_coco/' \
            'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_faster_rcnn(self):
        model_name = 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/' \
            'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_atss(self):
        model_name = 'atss/atss_r50_fpn_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'atss/atss_r50_fpn_1x_coco/' \
            'atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_fcos(self):
        model_name = 'fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = \
            'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/' \
            'mmdetection/v2.0/fcos/' \
            'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/' \
            'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco' \
            '_20200229-11f8c079.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_foveabox(self):
        model_name = 'foveabox/fovea_r50_fpn_4x4_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = 'http://download.openmmlab.com/mmdetection/v2.0/' \
            'foveabox/fovea_r50_fpn_4x4_1x_coco/' \
            'fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_dcn_faster_rcnn(self):
        model_name = 'dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = \
            'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/' \
            'mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/' \
            'faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_mask_rcnn(self):
        model_name = 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = \
            'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/' \
            'mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/' \
            'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
        opset_version = '11'
        cfg_options = {
            'model.test_cfg.rcnn.rescale_mask_to_input_shape': False
        }
        metrics = ('bbox', 'segm')

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version, cfg_options, metrics)

    def test_vfnet(self):
        # With config fix
        model_name = 'vfnet/vfnet_r50_fpn_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        # config_path = 'mmdetection/configs/' + model_name + '.py'
        checkpoint_url = \
            'https://openmmlab.oss-cn-hangzhou.aliyuncs.com/' \
            'mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/' \
            'vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth'
        cfg_options = {
            'data.test.pipeline.1.transforms.4.type': 'ImageToTensor',
            'data.test.pipeline.1.transforms.4.keys': ['img']
        }
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version, cfg_options)

    '''
    def test_cascade_rcnn(self):
        model_name = 'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = \
            'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/' \
            'mmdetection/v2.0/' \
            'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/' \
            'cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)
    def test_cornernet(self):
        model_name = 'cornernet/cornernet_hourglass104_mstest_10x5_210e_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = \
            'http://download.openmmlab.com/mmdetection/v2.0/' \
            'cornernet/cornernet_hourglass104_mstest_10x5_210e_coco/' \
            'cornernet_hourglass104_mstest_10x5_210e_coco_' \
            '20200824_185720-5fefbf1c.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)

    def test_efficientdet(self):
        model_name = 'efficientdet/retinanet_effd0_bifpn_1x_coco'
        config_path = osp.join(self.config_path_root, model_name + '.py')
        checkpoint_url = 'https://storage.openvinotoolkit.org/repositories/' \
            'mmdetection/models/efficientdet/' \
            'retinanet_effd0_bifpn_1x_coco/epoch_300.pth'
        opset_version = '11'

        self.run_export_test(model_name, config_path, checkpoint_url,
                             opset_version)
    '''


if __name__ == '__main__':
    # unittest.main()
    test_case = OpenvinoModelsTestCase()
    test_case.print_configuration_description()
    test_case.setUpClass()

    # Models from upstream with mods in mmcv and mmdet
    test_case.test_ssd()
    test_case.test_yolov3()
    test_case.test_fsaf()
    test_case.test_retinanet()
    test_case.test_faster_rcnn()
    test_case.test_fcos()
    test_case.test_mask_rcnn()

    # Successfully exported.
    test_case.test_foveabox()
    test_case.test_atss()
    test_case.test_vfnet()
    test_case.test_dcn_faster_rcnn()

    # Unsuccessfully exported.
    # test_cornernet() # need cummax
    # test_cascade_rcnn()
    # test_efficientdet()
