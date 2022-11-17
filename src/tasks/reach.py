import collections

import numpy as np
from dm_env import specs
from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.manipulation.shared import workspaces
from dm_control.manipulation.props import primitive
from dm_control.composer.variation import distributions
from dm_control.utils import rewards

from src import constants
from src.entities import Arena, Robotiq2f85, UR5e, cameras

_CTRL_LIMIT = .04
_TARGET_RADIUS = .05

_BOX_OFFSET = np.array([-.5, .05, 0.1])
_BOX_SIZE = constants.DEFAULT_BOX_HALFSIZE
_BOX_MASS = .3

_ReachWorkspace = collections.namedtuple(
    'ReachWorkspace', ['tcp_bbox', 'target_bbox', 'scene_bbox', 'arm_offset'])

_DEFAULT_WORKSPACE = _ReachWorkspace(
    tcp_bbox=workspaces.BoundingBox(
        _BOX_OFFSET + np.array([-.1, -.1, .2]),
        _BOX_OFFSET + np.array([.1, .1, .4])
    ),
    target_bbox=workspaces.BoundingBox(
        _BOX_OFFSET + np.array([-.1, -.1, 0]),
        _BOX_OFFSET + np.array([.1, .1, 0])
    ),
    scene_bbox=workspaces.BoundingBox(
        _BOX_OFFSET + np.array([-.2, -.2, -.1]),
        _BOX_OFFSET + np.array([.2, .2, .5])
    ),
    arm_offset=constants.ARM_OFFSET,
)


class ReachMocap(composer.Task):
    def __init__(self,
                 workspace: _ReachWorkspace = _DEFAULT_WORKSPACE,
                 control_timestep: float = constants.CONTROL_TIMESTEP,
                 img_size=(84, 84)
                 ):
        self._arena = Arena()
        self._arm = UR5e()
        self._arm.mjcf_model.actuator.remove()
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
            self._arena, cameras.KINECT, width=img_size[0], height=img_size[1]
        )

        self._prop = primitive.Box(half_lengths=_BOX_SIZE, mass=_BOX_MASS)
        self._prop.mjcf_model.find('geom', 'body_geom').rgba = (.6, 0, 0, 1)

        self._target = self._arena.add_free_entity(self._prop)
        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*workspace.target_bbox),
            quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=False,
            settle_physics=True,
        )
        
        worldbody = self.root_entity.mjcf_model.worldbody
        workspaces.add_target_site(
            body=self._prop.mjcf_model.worldbody,
            radius=_TARGET_RADIUS,
            visible=False,
            rgba=constants.RED,
            name='target_site'
        )
        workspaces.add_bbox_site(
            body=worldbody,
            lower=workspace.tcp_bbox.lower, upper=workspace.tcp_bbox.upper,
            rgba=constants.GREEN, name='tcp_spawn_area'
        )
        workspaces.add_bbox_site(
            body=worldbody,
            lower=workspace.target_bbox.lower,
            upper=workspace.target_bbox.upper,
            rgba=constants.BLUE, name='target_spawn_area'
        )

        self._arm.observables.enable_all()
        self._hand.observables.enable_all()
        self._arena.observables.enable_all()
        self._prop.observables.enable_all()
        # for v in self._task_observables.values():
        #     v.enabled = True

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def get_reward(self, physics):
        hand_pos = physics.bind(self._hand.tool_center_point).xpos
        target_pos = physics.bind(self._target).xpos
        distance = np.linalg.norm(hand_pos - target_pos)
        return rewards.tolerance(
            distance, bounds=(0, _TARGET_RADIUS), margin=_TARGET_RADIUS)

    def initialize_episode(self, physics, random_state):
        weld = physics.bind(self._weld)
        base = physics.bind(self._hand.base)
        tcp = physics.bind(self._hand.tool_center_point)
        eq_data = weld.data
        weld.active = 0

        joints = physics.bind(self._arm.joints)
        self._hand.set_grasp(physics, 1)
        joints.qpos = constants.HOME
        self._tcp_initializer(physics, random_state)
        self._prop_placer(physics, random_state)

        self._set_mocap(physics, tcp.xpos, base.xquat)
        eq_data[3:6] = -tcp.pos
        eq_data[6:10] = np.array([1, 0, 0, 0])
        eq_data[10] = 1
        weld.data = eq_data
        physics.forward()
        weld.active = 1

    def before_step(self, physics, action, random_state):
        pos, grip = action[:-1], action[-1]
        close_factor = (grip + 1) / 2.
        self._hand.set_grasp(physics, close_factor)
        mocap_pos, mocap_quat = self._get_mocap(physics)
        self._set_mocap(physics,
                        mocap_pos + _CTRL_LIMIT * pos,
                        mocap_quat)

    def action_spec(self, physics):
        lim = np.full((4,), 1, dtype=np.float32)
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
        pos = np.clip(pos, a_min=sbb.lower, a_max=sbb.upper)
        mocap.mocap_pos = pos
        mocap.mocap_quat = quat
