# retigate/baselines/__init__.py
# Add RAFT to exports

from .frame_diff import FrameDiffBaseline
from .mog2 import MOG2Baseline
from .disflow import DISFlowBaseline
from .farneback import FarnebackBaseline
from .raft_baseline import RAFTBaseline       

__all__ = [
    'FrameDiffBaseline',
    'MOG2Baseline',
    'DISFlowBaseline',
    'FarnebackBaseline',
    'RAFTBaseline',                       
]