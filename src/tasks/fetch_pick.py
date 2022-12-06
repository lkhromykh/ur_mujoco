# TODO: prune duplicates.
from typing import Tuple

import numpy as np
from dm_control.composer import initializers
from dm_control.manipulation.shared import workspaces
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions

from src import constants
from src.tasks import base
from src.tasks import lift

_BOX_SIZE = (.05, .03, .02)
_BOX_MASS = .1

_THRESHOLD = .005


class FetchPick(base.Task):
    """Closely resembles Lift task but require
    to fetch box in a precise position, not just height level."""
    def __init__(self,
                 workspace: lift._LiftWorkspace = lift.DEFAULT_WORKSPACE,
                 control_timestep: float = constants.CONTROL_TIMESTEP,
                 img_size: Tuple[int, int] = (84, 84),
                 threshold: float = _THRESHOLD
                 ):
        super().__init__(workspace, control_timestep, img_size)
        self._prop = lift.BoxWithVertexSites(
            half_lengths=_BOX_SIZE, mass=_BOX_MASS)
        self._prop.geom.rgba = (1, 0, 0, 1)
        self._arena.add_free_entity(self._prop)
        self._threshold = threshold

        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*workspace.prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=True,
            max_attempts_per_prop=60,
            max_settle_physics_attempts=20,
            max_settle_physics_time=5,
            settle_physics=True
        )

        self._target_site = workspaces.add_target_site(
            body=self.root_entity.mjcf_model.worldbody,
            radius=threshold,
            visible=False,
            rgba=constants.RED,
            name='target_site'
        )
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower, upper=workspace.prop_bbox.upper,
            rgba=constants.BLUE, name='prop_spawn_area')

        # Duct tape.
        self.eval_flag = False
        # Spoofing observables so they are included on _build.
        self._goal_img = np.zeros(img_size+(3,), np.uint8)
        self._goal_pos = np.zeros((3,), np.float64)
        self._task_observables['goal_image'] = \
            observable.Generic(lambda _: self._goal_img)
        self._task_observables['goal_pos'] = \
            observable.Generic(lambda _: self._goal_pos)

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
        super()._build_observables()
        self._prop.observables.enable_all()

    def initialize_episode_mjcf(self, random_state):
        if not self.eval_flag:
            self._mjcf_variation.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._initialize_midair(physics, random_state)
        self._prepare_goal(physics, random_state)
        physics.bind(self._target_site).pos = self._goal_pos

        if not self.eval_flag and random_state.choice([True, False]):
            self._initialize_midair(physics, random_state)
        else:
            super().initialize_episode(physics, random_state)
            self._prop_placer(physics, random_state)

    def get_reward(self, physics):
        pos, _ = self._prop.get_pose(physics)
        dist = np.linalg.norm(pos - self._goal_pos)
        return float(dist < self._threshold)

    # TODO: fix impossible configurations that lead to divergences.
    def _initialize_midair(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        mocap = physics.bind(self._mocap)
        physics.bind(self._hand.actuators[0]).ctrl = 255
        # set_pose only works because prop's parent is worldbody.
        self._prop.set_pose(physics, mocap.mocap_pos)
        xvel, qvel = map(
            lambda vel: np.zeros_like(vel.copy()),
            self._prop.get_velocity(physics))
        for _ in range(100):
            self._prop.set_pose(physics, mocap.mocap_pos)
            self._prop.set_velocity(physics, xvel, qvel)
            physics.step()

    def _prepare_goal(self, physics, random_state):
        img = self.task_observables['kinect/image']
        self._goal_img = img(physics, random_state).copy()
        pos, _ = self._prop.get_pose(physics)
        self._goal_pos = pos.copy()
