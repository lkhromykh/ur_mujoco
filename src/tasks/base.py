import abc
from typing import NamedTuple, Tuple

import numpy as np
from dm_env import specs
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer import initializers
from dm_control.manipulation.shared import workspaces
from dm_control.composer.variation import (
    colors, noises, distributions, variation_values
)

from src import constants
from src.entities import Arena, Robotiq2f85, UR5e, cameras


class RgbVariation(colors.RgbVariation):
    def __init__(self, low=0., high=1.):
        def d_fn(): return distributions.Uniform(low, high, single_sample=True)
        super().__init__(d_fn(), d_fn(), d_fn())

    def __call__(self,
                 initial_value=None,
                 current_value=None,
                 random_state=None
                 ):
        if current_value.size == 3:
            return variation_values.evaluate(
                [self._r, self._g, self._b],
                initial_value, current_value, random_state
            )
        return super().__call__(initial_value, current_value, random_state)


class BaseWorkspace(NamedTuple):
    tcp_bbox: workspaces.BoundingBox
    scene_bbox: workspaces.BoundingBox
    arm_offset: np.ndarray = constants.ARM_OFFSET


DEFAULT_SCENE_BBOX = workspaces.BoundingBox(
    lower=np.array([-.7, -.4, 0.]),
    upper=np.array([.1, .4, .8])
)


class Task(composer.Task, abc.ABC):

    def __init__(self,
                 workspace: BaseWorkspace,
                 control_timestep: float,
                 img_size: Tuple[int, int],
                 ):
        self._workspace = workspace
        self._control_timestep = control_timestep
        self._img_size = img_size

        self._arena = Arena()
        self._arm = UR5e()
        self._arm.mjcf_model.actuator.remove()
        self._hand = Robotiq2f85()
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self._mocap = self._arena.insert_mocap(self._hand.base)
        self._weld = self._arena.mjcf_model.find('equality', 'mocap_weld')

        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand, self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox)
        )

        self._task_observables, self._cameras = cameras.add_camera_observables(
            self._arena, cameras.KINECT, width=img_size[0], height=img_size[1]
        )
        self._kinect = self._cameras['kinect']
        # add gripper object detected obs?

        self._mjcf_variation = variation.MJCFVariator()
        self._physics_variation = variation.PhysicsVariator()
        self._built = False

    def __post_init__(self):
        self._build_observables()
        self._build_variations()
        self._built = True

    def _build_observables(self):
        """Decide what will agent observe."""
        self._arena.observables.enable_all()
        self._arm.observables.enable_all()
        self._hand.observables.enable_all()

        for obs in self._task_observables.values():
            obs.enabled = True

    def _build_variations(self):
        """Domain randomization goes here."""
        uni = distributions.Uniform

        self._mjcf_variation.bind_attributes(
            self._arena.skybox,
            rgb1=uni(),
            random=uni(high=.1)
        )
        self._mjcf_variation.bind_attributes(
            self._hand.tool_center_point,
            pos=noises.Multiplicative(uni(.8, 1.2))
        )
        for mat in self.root_entity.mjcf_model.asset.find_all('material'):
            self._mjcf_variation.bind_attributes(
                mat,
                rgba=RgbVariation(),
                specular=uni(high=.6),
                shininess=uni(high=.6),
                reflectance=uni(high=.1)
            )

        for light in self.root_entity.mjcf_model.worldbody.find_all('light'):
            self._mjcf_variation.bind_attributes(
                light,
                pos=noises.Additive(uni(-5, .5)),
                diffuse=uni(.3, .7),
                specular=uni(.1, .4),
                ambient=uni(high=.3)
            )
        for cam in self.root_entity.mjcf_model.worldbody.find_all('camera'):
            self._mjcf_variation.bind_attributes(
                cam,
                pos=noises.Additive(uni(-.1, .1)),
                xyaxes=noises.Multiplicative(uni(.7, 1.3)),
                fovy=noises.Additive(uni(-10, 10)),
            )

    def initialize_episode_mjcf(self, random_state):
        """Apply domain randomization and recompile model."""
        if not self._built:
            self.__post_init__()
        self._mjcf_variation.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        """Init scene.

        Arm and gripper init is done according to a task workspace.
        Task specific objects should be inited after.
        """
        self._physics_variation.apply_variations(physics, random_state)
        weld = physics.bind(self._weld)
        base = physics.bind(self._hand.base)
        joints = physics.bind(self._arm.joints)
        tcp = physics.bind(self._hand.tool_center_point)
        eq_data = weld.data
        weld.active = 0

        joints.qpos = constants.HOME
        self._tcp_initializer(physics, random_state)

        self._set_mocap(physics, tcp.xpos, base.xquat)
        eq_data[3:6] = -tcp.pos
        eq_data[6:10] = np.array([1, 0, 0, 0])
        eq_data[10] = 1
        weld.data = eq_data  # comment this
        physics.forward()
        weld.active = 1

    def before_step(self, physics, action, random_state):
        pos, grip = action[:-1], action[-1]
        close_factor = float(grip > 0)
        self._hand.set_grasp(physics, close_factor)
        mocap_pos, mocap_quat = self._get_mocap(physics)
        self._set_mocap(physics,
                        mocap_pos + constants.CTRL_LIMIT * pos,
                        mocap_quat)

    def action_spec(self, physics):
        lim = np.full((4,), 1, dtype=np.float32)
        return specs.BoundedArray(
            shape=lim.shape,
            dtype=lim.dtype,
            minimum=-lim,
            maximum=lim
        )

    def _get_mocap(self, physics):
        mocap = physics.bind(self._mocap)
        return mocap.mocap_pos, mocap.mocap_quat

    def _set_mocap(self, physics, pos, quat):
        """A mocap body pos is limited to a task bounding box."""
        mocap = physics.bind(self._mocap)
        sbb = self._workspace.scene_bbox
        pos = np.clip(pos, a_min=sbb.lower, a_max=sbb.upper)
        mocap.mocap_pos = pos
        mocap.mocap_quat = quat

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables
