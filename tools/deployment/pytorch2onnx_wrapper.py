import argparse


def parse_args_wrapper():
    parser = argparse.ArgumentParser(
        description='pytorch2onnx wrapper to handle additional parameters.')
    parser.add_argument(
        '--target',
        choices=['onnx', 'openvino'],
        default='openvino',
        help='Target model format.')

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


if __name__ == '__main__':
    wrapper_args, pytorch2onnx_args = parse_args()

    if wrapper_args.target == 'openvino':
        from workarounds.openvino import apply_all_fixes
        apply_all_fixes()
        from workarounds.mmdetection import apply_all_fixes
        apply_all_fixes()

    run_pytorch2onnx(pytorch2onnx_args)

    if wrapper_args.target == 'openvino':
        from workarounds.openvino import rename_input_onnx
        onnx_model_path = pytorch2onnx_args.output_file
        rename_input_onnx(onnx_model_path, 'input', 'image')
