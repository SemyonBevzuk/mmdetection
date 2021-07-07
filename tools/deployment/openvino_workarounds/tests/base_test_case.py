import os
import shutil
import sys
import unittest
import warnings
from functools import wraps

warnings.filterwarnings('ignore')


def remove_tmp_file(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        onnx_file = 'tmp.onnx'
        kwargs['onnx_file'] = onnx_file
        try:
            result = func(*args, **kwargs)
        finally:
            if os.path.exists(onnx_file):
                os.remove(onnx_file)
        return result

    return wrapper


class PatcheTestCaseBase(unittest.TestCase):
    model_filename = 'model'
    model_folder = os.path.join(os.path.dirname(__file__), 'models')
    old_path = ''
    old_meta_path = ''
    old_modules = {}
    '''
    def tearDown(self):
        import importlib
        for module in sys.modules.values():
            importlib.reload(module)
        #sys.modules.clear()
        sys.modules.update(self.old_modules)
        sys.meta_path = self.old_meta_path
        sys.path = self.old_path
    '''

    @classmethod
    def tearDownClass(cls):
        return
        shutil.rmtree(cls.model_folder)

    @classmethod
    def setUpClass(cls):
        # For import openvino_workarounds
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        if not os.path.exists(cls.model_folder):
            os.makedirs(cls.model_folder)
