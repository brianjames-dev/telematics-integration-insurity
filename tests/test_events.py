import numpy as np
from src.processing.feature_defs import flag_harsh_brake, flag_harsh_accel, flag_corner

def test_event_flags_simple():
    accel = np.array([0.0, -3.5, -2.0, 3.0, 2.6, 0.0])
    gyro = np.array([0.1, 0.4, -0.5, 0.2, -0.1, 0.0])

    assert flag_harsh_brake(accel).sum() == 1
    assert flag_harsh_accel(accel).sum() == 2
    assert flag_corner(gyro).sum() == 2
