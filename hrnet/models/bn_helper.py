import torch
import functools

from ..config import config

if len(config.GPUS) == 1 and config.LOCAL_RANK < 0:
    BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
    relu_inplace = False
else:
    if torch.__version__.startswith('0') or config.LOCAL_RANK < 0:
        from .sync_bn.inplace_abn.bn import InPlaceABNSync
        BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
        BatchNorm2d_class = InPlaceABNSync
    else:
        BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
        relu_inplace = True
