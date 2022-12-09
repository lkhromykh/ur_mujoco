from typing import NamedTuple, Tuple
import itertools

import numpy as np
from dm_control.composer import initializers
from dm_control.manipulation.shared import workspaces
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions

from src import constants
from src.tasks import base
from src.entities import props

_DISTANCE_TO_LIFT = .1
_BOX_OFFSET = np.array([-.5, .05, .1])
_BOX_SIZE = (.05, .03, .02)
_BOX_MASS = .1


class _LiftWorkspace(NamedTuple):
    tcp_bbox: workspaces.BoundingBox
    scene_bbox: workspaces.BoundingBox
    arm_offset: np.ndarray
    prop_bbox: workspaces.BoundingBox


DEFAULT_WORKSPACE = _LiftWorkspace(
    tcp_bbox=workspaces.BoundingBox(
        lower=_BOX_OFFSET + np.array([-.1, -.1, .2]),
        upper=_BOX_OFFSET + np.array([.1, .1, .4])
    ),
    scene_bbox=base.DEFAULT_SCENE_BBOX,
    prop_bbox=workspaces.BoundingBox(
        lower=_BOX_OFFSET + np.array([-.1, -.1, 0.]),
        upper=_BOX_OFFSET + np.array([.1, .1, 0.]),
    ),
    arm_offset=constants.ARM_OFFSET
)


class Lift(base.Task):
    def __init__(self,
                 workspace: _LiftWorkspace = DEFAULT_WORKSPACE,
                 control_timestep: float = constants.CONTROL_TIMESTEP,
                 img_size: Tuple[int, int] = (84, 84),
                 target_height: float = _DISTANCE_TO_LIFT
                 ):
        super().__init__(workspace, control_timestep, img_size)
        self._prop = props.BoxWithVertexSites(
            half_lengths=_BOX_SIZE, mass=_BOX_MASS)
        self._prop.geom.rgba = (1, 0, 0, 1)
        self._arena.add_free_entity(self._prop)
        self._target_height = target_height
        self._episode_height = None

        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*workspace.prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            max_settle_physics_attempts=5,
            ignore_collisions=False,
            settle_physics=True
        )

        self._target_height_site = workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=(-1, -1, 0), upper=(1, 1, 0),
            rgba=constants.RED, name='target_height')
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower, upper=workspace.prop_bbox.upper,
            rgba=constants.BLUE, name='prop_spawn_area')

        self.__post_init__()

    def _build_variations(self):
        super()._build_variations()
        self._mjcf_variation.bind_attributes(
            self._prop.geom,
            rgba=base.RgbVariation(),
            size=distributions.Uniform(.01, .035),
            mass=distributions.Uniform(.05, .3)
        )

    def _build_observables(self):
        # Spoofing observations so they are included in the env.observation_spec
        self._task_observables['goal_image'] =\
            self._task_observables['kinect/image']
        self._task_observables['goal_pos'] = observable.MJCFFeature(
            'pos', self._prop.geom)
        super()._build_observables()
        self._prop.observables.enable_all()

    def initialize_episode(self, physics, random_state):
        self._place_prop_in_hand(physics, random_state)
        self._prepare_goal(physics, random_state)
        self._prop.set_velocity(physics, np.zeros(3), np.zeros(3))

        lifted_start = random_state.choice(2)
        if lifted_start:
            self._place_prop_in_hand(physics, random_state)
            self._episode_height = self._target_height
        else:
            super().initialize_episode(physics, random_state)
            self._prop_placer(physics, random_state)
            initial_prop_height = self._get_height_of_lowest_vertex(physics)
            self._episode_height = self._target_height + initial_prop_height
        physics.bind(self._target_height_site).pos[2] = self._episode_height

    def _get_height_of_lowest_vertex(self, physics):
        return min(physics.bind(self._prop.vertices).xpos[:, 2])

    def get_reward(self, physics):
        prop_height = self._get_height_of_lowest_vertex(physics)
        return 5 * float(prop_height > self._target_height)

    # TODO: fix impossible configurations that lead to divergences.
    def _place_prop_in_hand(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        mocap = physics.bind(self._mocap)
        act = physics.bind(self._hand.actuators[0])
        self._prop.set_pose(physics, mocap.mocap_pos)
        xvel, qvel = map(
            lambda vel: np.zeros_like(vel.copy()),
            self._prop.get_velocity(physics))
        act.ctrl = 255
        for _ in range(100):
            self._prop.set_pose(physics, mocap.mocap_pos)
            self._prop.set_velocity(physics, xvel, qvel)
            physics.step()

    def _prepare_goal(self, physics, random_state):
        img = self.task_observables['kinect/image']
        self._goal_img = img(physics, random_state).copy()
        self._goal_pos = physics.bind(self._mocap).mocap_pos.copy()
        goal_img = observable.Generic(lambda _: self._goal_img)
        goal_pos = observable.Generic(lambda _: self._goal_pos)
        goal_img.enabled = True
        goal_pos.enabled = True
        self._task_observables['goal_image'] = goal_img
        self._task_observables['goal_pos'] = goal_pos
