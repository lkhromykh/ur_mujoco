import numpy as np

# Predefined RGBA values
RED = (1., 0., 0., 0.3)
GREEN = (0., 1., 0., 0.3)
BLUE = (0., 0., 1., 0.3)
CYAN = (0., 1., 1., 0.3)
MAGENTA = (1., 0., 1., 0.3)
YELLOW = (1., 1., 0., 0.3)

TASK_SITE_GROUP = 3

PHYSICS_TIMESTEP = .005
CONTROL_TIMESTEP = .1

ARM_OFFSET = (0, 0.4, 0)
HOME = np.array([np.pi, -1.5708,  1.5708, -1.5708, -1.5708,  0.])
DEFAULT_BOX_HALFSIZE = (.05, .03, .02)

DOWN_QUATERNION = np.array([0., 0.70710678118, 0.70710678118, 0.])
