"""dm_control/manipulation/shared/cameras.py"""
import collections

from dm_control import composer
from dm_control.composer.observation import observable

CameraSpec = collections.namedtuple('CameraSpec', ['name', 'pos', 'xyaxes'])

FRONT_CLOSE = CameraSpec(
    name='front_close',
    pos=(0., -0.6, 0.75),
    xyaxes=(1., 0., 0., 0., 0.7, 0.75)
)

FRONT_FAR = CameraSpec(
    name='front_far',
    pos=(0., -0.8, 1.),
    xyaxes=(1., 0., 0., 0., 0.7, 0.75)
)

TOP_DOWN = CameraSpec(
    name='top_down',
    pos=(0., 0., 2.5),
    xyaxes=(1., 0., 0., 0., 1., 0.)
)

LEFT_CLOSE = CameraSpec(
    name='left_close',
    pos=(-0.6, 0., 0.75),
    xyaxes=(0., -1., 0., 0.7, 0., 0.75)
)


RIGHT_CLOSE = CameraSpec(
    name='right_close',
    pos=(0.6, 0., 0.75),
    xyaxes=(0., 1., 0., -0.7, 0., 0.75)
)


def add_camera_observables(entity: composer.Entity,
                           *camera_specs: CameraSpec,
                           **kwargs
                           ) -> collections.OrderedDict:
    obs_dict = collections.OrderedDict()
    for spec in camera_specs:
        camera = entity.mjcf_model.worldbody.add('camera', **spec._asdict())
        obs = observable.MJCFCamera(camera, **kwargs)
        obs_dict[spec.name] = obs
    return obs_dict
