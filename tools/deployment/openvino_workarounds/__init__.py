import os
'''
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL
is incompatible with libgomp.so.1 library.
    Try to import numpy first or set the threading layer accordingly.
    Set MKL_SERVICE_FORCE_INTEL to force it.
'''
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from .symbolic import register_extra_symbolics_for_openvino
from .mmdetection import update_default_args_value, \
                         apply_all_fixes as apply_all_fixes_mmdetection
from .openvino import rename_input_onnx, \
                      apply_all_fixes as apply_all_fixes_openvino

__all__ = [
    'register_extra_symbolics_for_openvino',
    'update_default_args_value', 'apply_all_fixes_mmdetection',
    'rename_input_onnx', 'apply_all_fixes_openvino'
]
