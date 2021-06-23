import inspect
import re
import onnx
from functools import partial


def update_default_args_value(func, **updated_args):
    new_func = partial(func, **updated_args)
    return new_func


class OpenvinoExportHelper():
    """
    This class contains useful methods that are needed to create OpenVINO
    compatible models.
    """

    @staticmethod
    def process_extra_symbolics_for_openvino(opset=11):
        """
        Registers additional symbolic functions for OpenVINO (defined in
        'symbolic.py') and applies replacement to the original symbolic
        functions.
        """
        assert opset >= 10
        from . import symbolic
        function_members = inspect.getmembers(symbolic, inspect.isfunction)
        pattern = 'get_patch_'
        patch_functions = []
        for name, func in function_members:
            if re.match(pattern, name):
                patch_functions.append(func)

        domain = 'mmdet_custom'
        from torch.onnx.symbolic_registry import register_op
        for patch_func in patch_functions:
            patch = patch_func()
            opname = patch.get_operation_name()
            symbolic_function = patch.get_symbolic_func()
            register_op(opname, symbolic_function, domain, opset)
            patch.apply_patch()

    @staticmethod
    def rename_input_onnx(onnx_model_path, old_name, new_name):
        """
        Changes the input name of the model.

        Useful for use in tests from OTEDetection.
        """
        onnx_model = onnx.load(onnx_model_path)
        for node in onnx_model.graph.node:
            for i in range(len(node.input)):
                if node.input[i] == old_name:
                    node.input[i] = new_name

        for input in onnx_model.graph.input:
            if input.name == old_name:
                input.name = new_name

        onnx.save(onnx_model, onnx_model_path)
