import numpy as np

# Predefined RGBA values
RED = (1., 0., 0., 0.3)
GREEN = (0., 1., 0., 0.3)
BLUE = (0., 0., 1., 0.3)
CYAN = (0., 1., 1., 0.3)
MAGENTA = (1., 0., 1., 0.3)
YELLOW = (1., 1., 0., 0.3)

TASK_SITE_GROUP = 3
SENSOR_SITE_GROUP = 4
MOCAP_SITE_GROUP = 5

PHYSICS_TIMESTEP = .002
CONTROL_TIMESTEP = .5

CTRL_LIMIT = .05
TIME_LIMIT = 30

ARM_OFFSET = (0, 0, 0)
HOME = np.array([1.5708, -1.5708,  1.5708, -1.5708, -1.5708,  0.])

DOWN_QUATERNION = np.array([0., 0.70710678118, 0.70710678118, 0.])
