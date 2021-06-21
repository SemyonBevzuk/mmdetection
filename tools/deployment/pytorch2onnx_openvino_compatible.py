import argparse
import openvino_workarounds as workarounds


def parse_args_wrapper(args_list=None):
    parser = argparse.ArgumentParser(
        description='pytorch2onnx wrapper to handle additional parameters.')
    parser.add_argument(
        '--not_strip_doc_string',
        action='store_true',
        help='If is, does not strip the field “doc_string”'
        'from the exported model, which information about the stack trace.')

    args, other_args_list = parser.parse_known_args(args=args_list)
    return args, other_args_list


def parse_args():
    wrapper_args, args_list = parse_args_wrapper()
    from pytorch2onnx import parse_args as parse_args_pytorch2onnx
    pytorch2onnx_args = parse_args_pytorch2onnx(args_list)
    return wrapper_args, pytorch2onnx_args


def run_pytorch2onnx(args):
    from pytorch2onnx import main
    main(args)


def update_strip_doc_string():
    from torch import onnx
    onnx.export = workarounds.update_default_args_value(
        onnx.export, strip_doc_string=False)


if __name__ == '__main__':
    wrapper_args, pytorch2onnx_args = parse_args()

    workarounds.apply_all_fixes_openvino()
    workarounds.apply_all_fixes_mmdetection()

    if wrapper_args.not_strip_doc_string:
        update_strip_doc_string()

    workarounds.register_extra_symbolics_for_openvino(
        pytorch2onnx_args.opset_version)

    run_pytorch2onnx(pytorch2onnx_args)

    onnx_model_path = pytorch2onnx_args.output_file
    workarounds.rename_input_onnx(onnx_model_path, 'input', 'image')
