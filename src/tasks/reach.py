import collections
from typing import Optional, Iterable, NamedTuple

import numpy as np
from dm_env import specs
from dm_control import mjcf
from dm_control import composer
from dm_control.composer import observation
from dm_control.composer import initializers
from dm_control.manipulation.shared import workspaces
from dm_control.composer.variation import distributions

from src import constants
from src.entities import Arena, Robotiq2f85, UR5e, primitive, cameras

_CTRL_LIMIT = .04

ReachWorkspace = collections.namedtuple(
    'ReachWorkspace', ["tcp_bbox", "target_bbox", "scene_bbox", "arm_offset"])

_PROP_OFFSET = .04
_DEFAULT_WORKSPACE = ReachWorkspace(
    tcp_bbox=workspaces.BoundingBox(
        np.array([-.1, -.1, .2]),
        np.array([.1, .1, .4])
    ),
    target_bbox=workspaces.BoundingBox(
        np.array([-.1, -.1, _PROP_OFFSET]),
        np.array([.1, .1, _PROP_OFFSET])
    ),
    scene_bbox=workspaces.BoundingBox(
        np.array([-.2, -.2, 0]),
        np.array([.2, .2, .4])
    ),
    arm_offset=constants.ARM_OFFSET,
)

_THRESHOLD = .05


class Reach(composer.Task):
    def __init__(self,
                 workspace: ReachWorkspace = _DEFAULT_WORKSPACE,
                 control_timestep: float = constants.CONTROL_TIMESTEP
                 ):
        self._arena = Arena()
        self._arm = UR5e()
        self._hand = Robotiq2f85()
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._workspace = workspace

        self._mocap = self._arena.insert_mocap(self._hand.base)
        self._weld = self._arena.mjcf_model.find('equality', 'mocap_weld')

        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand, self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox)
        )
        self._task_observables = cameras.add_camera_observables(
            self._arena, cameras.FRONT_CLOSE, width=64, height=64
        )

        self._prop = primitive.Box(half_lengths=constants.DEFAULT_BOX_HALFSIZE,
                                   mass=.3)
        self._prop.mjcf_model.find('geom', 'body_geom').rgba = (.5, 0, 0, 1)

        self._target = self._arena.add_free_entity(self._prop)
        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*workspace.target_bbox),
            quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=False,
            settle_physics=True,
        )

        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.tcp_bbox.lower, upper=workspace.tcp_bbox.upper,
            rgba=constants.GREEN, name='tcp_spawn_area'
        )
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.target_bbox.lower,
            upper=workspace.target_bbox.upper,
            rgba=constants.BLUE, name='target_spawn_area'
        )

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def get_reward(self, physics):
        target_pos, target_quat = self._prop.get_pose(physics)
        tcp_pos, tcp_quat = self._get_mocap(physics)
        dist = np.linalg.norm(target_pos - tcp_pos)
        return dist < _THRESHOLD

    def initialize_episode(self, physics, random_state):
        weld = physics.bind(self._weld)
        base = physics.bind(self._hand.base)
        tcp = physics.bind(self._hand.tool_center_point)

        weld.active = 0
        physics.bind(self._arm.joints).qpos = constants.HOME
        self._tcp_initializer(physics, random_state)
        self._prop_placer(physics, random_state)

        self._set_mocap(physics, base.xpos, base.xquat)
        eq_data = np.zeros((11,))
        eq_data[:3] = tcp.pos
        eq_data[3] = 1
        weld.data = eq_data
        weld.active = 1
        physics.forward()

    def before_step(self, physics, action, random_state):
        pos, quat, grip = np.split(action, [3, 7], -1)
        close_factor = (grip + 1) / 2.
        self._hand.set_grasp(physics, close_factor)
        mocap = physics.bind(self._mocap)
        self._set_mocap(physics,
                        mocap.mocap_pos + _CTRL_LIMIT * pos,
                        mocap.mocap_quat + _CTRL_LIMIT * quat)

    def action_spec(self, physics):
        lim = np.full((8,), 1)
        return specs.BoundedArray(
            shape=lim.shape,
            dtype=lim.dtype,
            minimum=-lim,
            maximum=lim
        )

    @property
    def mocap(self):
        return self._mocap

    @property
    def weld(self):
        return self._weld

    def _get_mocap(self, physics):
        mocap = physics.bind(self.mocap)
        return mocap.mocap_pos, mocap.mocap_quat

    def _set_mocap(self, physics, pos, quat):
        mocap = physics.bind(self.mocap)
        sbb = self._workspace.scene_bbox
        # pos = np.clip(pos, a_min=sbb.lower, a_max=sbb.upper)
        mocap.mocap_pos = pos
        mocap.mocap_quat = quat
