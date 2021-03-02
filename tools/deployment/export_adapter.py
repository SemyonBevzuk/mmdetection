import argparse

from pytorch2onnx import main as pytorch2onnx_main
from pytorch2onnx import parse_args as pytorch2onnx_parse_args

#import torch



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX or to OpenVino')
    parser.add_argument('--target', type=str, default='onnx', choices=['onnx', 'openvino'],
        help='Target model format. When "openvino" is selected, additional symbolics will be used. Selecting "onnx" will use the default export.')
    args = parser.parse_known_args()
    return args


def change_arange_to_opset9(g, *args):
    from torch.onnx.symbolic_opset9 import arange
    return arange(g, *args)


def register_openvino_symbolics(opset=11):
    from torch.onnx.symbolic_registry import register_op
    register_op('arange', change_arange_to_opset9, '', opset)


if __name__ == '__main__':
    adapter_args, args = parse_args()

    if adapter_args.target == 'openvino':
        register_openvino_symbolics()

    pytorch2onnx_main(pytorch2onnx_parse_args(args))
