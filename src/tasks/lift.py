import itertools
import collections

import numpy as np
from dm_env import specs
from dm_control import composer
from dm_control.composer import observation
from dm_control.composer import initializers
from dm_control.manipulation.props import primitive
from dm_control.manipulation.shared import workspaces
from dm_control.composer.variation import distributions

from src import constants
from src.entities import Arena, Robotiq2f85, UR5e, cameras

_LiftWorkspace = collections.namedtuple(
    'LiftWorkspace', ['prop_bbox', 'tcp_bbox', 'arm_offset'])


_DISTANCE_TO_LIFT = .1
_BOX_SIZE = (.05, .03, .02)
_BOX_MASS = .3
_CTRL_LIM = .04
_DEFAULT_WORKSPACE = _LiftWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-.1, -.1, .02), upper=(.1, .1, .05)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-.1, -.1, .2), upper=(.1, .1, .4)),
    arm_offset=constants.ARM_OFFSET
)


class _VertexSitesMixin:
    def _add_vertex_sites(self, box_geom_or_site):
        """Add sites corresponding to the vertices of a box geom or site."""
        offsets = (
            (-half_length, half_length) for half_length in box_geom_or_site.size)
        site_positions = np.vstack(itertools.product(*offsets))
        if box_geom_or_site.pos is not None:
            site_positions += box_geom_or_site.pos
        self._vertices = []
        for i, pos in enumerate(site_positions):
            site = box_geom_or_site.parent.add(
                'site', name='vertex_' + str(i),
                pos=pos, type='sphere', size=[0.002],
                rgba=constants.RED, group=constants.TASK_SITE_GROUP)
            self._vertices.append(site)
    
    @property
    def vertices(self):
        return self._vertices


class _BoxWithVertexSites(primitive.Box, _VertexSitesMixin):
    """Subclass of `Box` with sites marking the vertices of the box geom."""

    def _build(self, *args, **kwargs):
        super()._build(*args, **kwargs)
        self._add_vertex_sites(self.geom)


class Lift(composer.Task):
    def __init__(self,
                 workspace: _LiftWorkspace = _DEFAULT_WORKSPACE,
                 control_timestep: float = constants.CONTROL_TIMESTEP
                 ):
        self._arena = Arena()
        self._arm = UR5e()
        self._hand = Robotiq2f85()
        _attach_hand_to_arm(self._arm, self._hand)
        self._arena.attach_offset(self._arm, workspace.arm_offset)

        self._prop = _BoxWithVertexSites(half_lengths=_BOX_SIZE, mass=_BOX_MASS)
        self._prop.mjcf_model.find('geom', 'body_geom').rgba = (1, 0, 0, 1)
        self._arena.add_free_entity(self._prop)

        self._arm.observables.enable_all()
        self._hand.observables.enable_all()
        self._arena.observables.enable_all()
        self._prop.observables.enable_all()

        self.control_timestep = control_timestep
        self._task_observables = cameras.add_camera_observables(
            self._arena, cameras.FRONT_CLOSE, height=64, width=64
        )
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand, self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox),
            quaternion=np.array([1, 0, 0, 0.]),
        )

        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*workspace.prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=True,
            settle_physics=True
        )

        self._target_height_site = workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=(-1, -1, 0), upper=(1, 1, 0),
            rgba=constants.RED, name='target_height')
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.tcp_bbox.lower, upper=workspace.tcp_bbox.upper,
            rgba=constants.GREEN, name='tcp_spawn_area')
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower, upper=workspace.prop_bbox.upper,
            rgba=constants.BLUE, name='prop_spawn_area')

        def object_detected(physics):
            is_detected = np.atleast_1d(0 < physics.data.ten_length < .76)
            return is_detected.astype(np.float32)
        self._task_observables['ur5e/robotiq_2f85/object_detected'] =\
            observation.observable.Generic(object_detected)

        for value in self._task_observables.values():
            value.enabled = True

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode(self, physics, random_state):
        # self._hand.set_grasp(physics, 0)
        physics.bind(self._arm.joints).qpos = constants.HOME
        self._prop_placer(physics, random_state)
        self._tcp_initializer(physics, random_state)
        initial_prop_height = self._get_height_of_lowest_vertex(physics)
        self._target_height = _DISTANCE_TO_LIFT + initial_prop_height
        physics.bind(self._target_height_site).pos[2] = self._target_height

    def before_step(self, physics, action, random_state):
        low, high = physics.model.actuator_ctrlrange[:-1].T
        qpos = physics.bind(self._arm.joints).qpos
        arm, hand = np.split(action, [6])
        hand = 255 if hand > 0 else 0
        arm = np.clip(_CTRL_LIM * arm + qpos, a_min=low, a_max=high)
        ctrl = np.concatenate((arm, [hand]))
        physics.set_control(ctrl)

    def action_spec(self, physics):
        shape = (len(physics.model.actuator_ctrlrange),)
        return specs.BoundedArray(
            minimum=np.full(shape, -1.),
            maximum=np.full(shape, 1.),
            shape=shape,
            dtype=np.float32
        )

    @property
    def arm(self):
        return self._arm

    @property
    def hand(self):
        return self._hand

    @property
    def task_observables(self):
        return self._task_observables

    def _get_height_of_lowest_vertex(self, physics):
        return min(physics.bind(self._prop.vertices).xpos[:, 2])

    def get_reward(self, physics):
        prop_height = self._get_height_of_lowest_vertex(physics)
        return float(prop_height > self._target_height)


def _attach_hand_to_arm(arm: composer.Entity, hand: composer.Entity):
    arm_mjcf = arm.mjcf_model
    hand_mjcf = hand.mjcf_model

    nu = len(hand_mjcf.find_all('actuator'))
    nq = len(hand.mjcf_model.find_all('joint'))

    kf = arm_mjcf.find('key', 'home')
    if kf is not None:
        kf.ctrl = np.concatenate([kf.ctrl, np.zeros(nu)])
        kf.qpos = np.concatenate([kf.qpos, np.zeros(nq)])

    arm.attach(hand)
