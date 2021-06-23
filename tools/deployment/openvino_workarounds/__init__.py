import os
'''
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL
is incompatible with libgomp.so.1 library.
    Try to import numpy first or set the threading layer accordingly.
    Set MKL_SERVICE_FORCE_INTEL to force it.
'''
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from .mmdetection import apply_all_fixes as apply_all_fixes_mmdetection
from .openvino import apply_all_fixes as apply_all_fixes_openvino
from .openvino_helper import OpenvinoExportHelper, update_default_args_value


__all__ = [
    'OpenvinoExportHelper', 'update_default_args_value',
    'apply_all_fixes_mmdetection', 'apply_all_fixes_openvino'
]
