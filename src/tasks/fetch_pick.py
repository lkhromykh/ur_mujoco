# TODO: remove duplication.
from typing import Tuple

import numpy as np
from dm_control.composer import initializers
from dm_control.manipulation.shared import workspaces
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions

from src import constants
from src.tasks import base
from src.tasks import lift

_DISTANCE_TO_LIFT = .1
_BOX_OFFSET = np.array([-.5, .05, .1])
_BOX_SIZE = (.05, .03, .02)
_BOX_MASS = .1

_EASY_THRESHOLD = .05
_HARD_THRESHOLD = .01


class FetchPick(base.Task):
    """Closely resembles Lift task but require
    to fetch box in a precise position not just height level."""
    def __init__(self,
                 workspace: lift._LiftWorkspace = lift.DEFAULT_WORKSPACE,
                 control_timestep: float = constants.CONTROL_TIMESTEP,
                 img_size: Tuple[int, int] = (84, 84),
                 threshold_dist: float = _EASY_THRESHOLD
                 ):
        super().__init__(workspace, control_timestep, img_size)
        self._prop = lift.BoxWithVertexSites(
            half_lengths=_BOX_SIZE, mass=_BOX_MASS)
        self._prop.geom.rgba = (1, 0, 0, 1)
        self._arena.add_free_entity(self._prop)
        self._threshold = threshold_dist

        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*workspace.prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            max_settle_physics_attempts=5,
            ignore_collisions=False,
            settle_physics=True
        )

        self._target_site = workspaces.add_target_site(
            body=self.root_entity.mjcf_model.worldbody,
            radius=threshold_dist,
            visible=False,
            rgba=constants.RED,
            name='target_distance'
        )
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
        physics.bind(self._target_site).pos = self._goal_pos
        self._prop.set_velocity(physics, np.zeros(3), np.zeros(3))

        lifted_start = random_state.choice(2)
        if lifted_start:
            self._place_prop_in_hand(physics, random_state)
        else:
            super().initialize_episode(physics, random_state)
            self._prop_placer(physics, random_state)

    def get_reward(self, physics):
        pos, _ = self._prop.get_pose(physics)
        dist = np.linalg.norm(pos - self._goal_pos)
        return float(dist < self._threshold)

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
