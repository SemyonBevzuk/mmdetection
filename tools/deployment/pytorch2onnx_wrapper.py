import argparse


def parse_args_wrapper():
    parser = argparse.ArgumentParser(
        description='pytorch2onnx wrapper to handle additional parameters.')
    parser.add_argument(
        '--target',
        choices=['onnx', 'openvino'],
        default='openvino',
        help='Target model format.')
    parser.add_argument(
        '--not_strip_doc_string',
        action='store_true',
        help='If is, does not strip the field “doc_string”'
        'from the exported model, which information about the stack trace.'
    )

    args, other_args_list = parser.parse_known_args()
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
    from workarounds.mmdetection import update_default_arg_value
    update_default_arg_value(onnx.export, 'strip_doc_string', False)


if __name__ == '__main__':
    wrapper_args, pytorch2onnx_args = parse_args()

    if wrapper_args.target == 'openvino':
        from workarounds.openvino import apply_all_fixes
        apply_all_fixes()
        from workarounds.mmdetection import apply_all_fixes
        apply_all_fixes()

    if wrapper_args.not_strip_doc_string:
        update_strip_doc_string()

    run_pytorch2onnx(pytorch2onnx_args)

    if wrapper_args.target == 'openvino':
        from workarounds.openvino import rename_input_onnx
        onnx_model_path = pytorch2onnx_args.output_file
        rename_input_onnx(onnx_model_path, 'input', 'image')
