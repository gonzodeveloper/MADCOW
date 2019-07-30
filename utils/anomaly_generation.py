import numpy as np


def sensor_failure(window, sensor_idx, begin_timestep):

    distorted = np.array(window)
    distorted[begin_timestep::, sensor_idx] = 0
    return distorted


def sensor_offset(window, sensor_idx, begin_timestep, offset):
    distorted = np.array(window)
    distorted[begin_timestep::, sensor_idx] *= offset
    return distorted


def sensor_drift(window, sensor_idx, begin_timestep, offset_step):
    distorted = np.array(window)
    prev_offset = 1
    for timestep in distorted[begin_timestep:]:
        timestep[sensor_idx] *= (prev_offset + offset_step)
        prev_offset += offset_step
    return distorted
