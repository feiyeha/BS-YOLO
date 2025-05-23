# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""

from .OutlookAttention import *
from .MSCA import *
from .Moganet import *
from .WTConv import WTConv2d
from .condconv import CondConv2D
from .ELA import ELA
from .FocalModulation import FocalModulation
from .common import *