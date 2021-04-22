from functools import partial

import numpy as np
import torch
from six.moves import map, zip

from ..mask.structures import BitmapMasks, PolygonMasks


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def mask2ndarray(mask):
    """Convert Mask to ndarray..

    Args:
        mask (:obj:`BitmapMasks` or :obj:`PolygonMasks` or
        torch.Tensor or np.ndarray): The mask to be converted.

    Returns:
        np.ndarray: Ndarray mask of shape (n, h, w) that has been converted
    """
    if isinstance(mask, (BitmapMasks, PolygonMasks)):
        mask = mask.to_ndarray()
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        raise TypeError(f'Unsupported {type(mask)} data type')
    return mask


def dummy_pad(x, padding):

    class DummyPad(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, padding):
            return torch.nn.functional.pad(x, padding)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = None
            if ctx.needs_input_grad[0]:
                grad_output = grad_input
            return grad_output, None

        @staticmethod
        def symbolic(g, x, padding):
            return g.op('Identity', x)

    return DummyPad.apply(x, padding)


def arange(start=0,
           end=None,
           step=1,
           out=None,
           dtype=None,
           layout=torch.strided,
           device=None,
           requires_grad=False):
    if torch.onnx.is_in_onnx_export():
        if end is None:
            raise ValueError('End of range must be defined.')
        assert out is None
        assert layout == torch.strided

        start_tensor = torch.as_tensor(start, dtype=torch.long, device=device)
        end_tensor = torch.as_tensor(end, dtype=torch.long, device=device)
        n = end_tensor - start_tensor

        if isinstance(step, torch.Tensor):
            n = (n + step - 1) // step

        result = torch.ones(n, layout=layout, device=device)\
            .nonzero().view(-1) + start_tensor

        if isinstance(step, torch.Tensor) or step > 1:
            result = result.view(-1, step)\
                .index_select(1, torch.zeros(1, dtype=torch.long))\
                .view(-1)

        if dtype is not None:
            result = result.to(dtype)

        return result
    else:
        return torch.arange(
            start=start,
            end=end,
            step=step,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad)
