"""Split a trip into continuous segments at time gaps."""
import numpy as np

MAX_GAP_S = 60.0       # longer than this between two fixes starts a new segment
MAX_SPEED_MPS = 80.0   # implied speed above this is a jump, not driving


def segment_ids(t, xy, max_gap_s=MAX_GAP_S, max_speed_mps=MAX_SPEED_MPS):
    dt = np.diff(t)
    step = np.hypot(*np.diff(xy, axis=0).T)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.inf)
    brk = (dt > max_gap_s) | (dt <= 0) | (speed > max_speed_mps)
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int32)
