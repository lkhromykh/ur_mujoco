# TODO: prune duplicates.
from typing import Tuple, NamedTuple

import numpy as np
from dm_control.composer import initializers
from dm_control.rl.control import PhysicsError
from dm_control.manipulation.shared import workspaces
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions

from src import constants
from src.tasks import base
from src.tasks import lift

_BOX_SIZE = (.05, .03, .02)
_BOX_OFFSET = np.array([-.5, .05, .1])
_BOX_MASS = .005

_THRESHOLD = .008


class FetchWorkspace(NamedTuple):
    tcp_bbox: workspaces.BoundingBox
    scene_bbox: workspaces.BoundingBox
    arm_offset: np.ndarray
    prop_bbox: workspaces.BoundingBox


DEFAULT_WORKSPACE = FetchWorkspace(
    tcp_bbox=workspaces.BoundingBox(
        lower=_BOX_OFFSET + np.array([-.15, -.15, .1]),
        upper=_BOX_OFFSET + np.array([.15, .15, .3])
    ),
    scene_bbox=base.DEFAULT_SCENE_BBOX,
    prop_bbox=workspaces.BoundingBox(
        lower=_BOX_OFFSET + np.array([-.15, -.15, 0.]),
        upper=_BOX_OFFSET + np.array([.15, .15, 0.]),
    ),
    arm_offset=constants.ARM_OFFSET
)


class FetchPick(base.Task):
    """Closely resembles Lift task but require
    to fetch box in a precise position, not just height level."""
    def __init__(self,
                 workspace: FetchWorkspace = DEFAULT_WORKSPACE,
                 control_timestep: float = constants.CONTROL_TIMESTEP,
                 img_size: Tuple[int, int] = (84, 84),
                 threshold: float = _THRESHOLD,
                 ):
        super().__init__(workspace, control_timestep, img_size)
        self._prop = lift.BoxWithVertexSites(
            half_lengths=_BOX_SIZE, mass=_BOX_MASS)
        self._prop.geom.rgba = (1, 0, 0, 1)
        self._arena.add_free_entity(self._prop)
        self._threshold = threshold

        tbb = workspace.tcp_bbox
        self._tcp_center = (tbb.lower + tbb.upper) / 2

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

    def initialize_episode(self, physics, random_state):
        # Sample grounded goal: goal is on the table.
        # And sample midair init: prop is grasped by the gripper.
        # Eval should never be grounded nor midair.
        try:
            grounded_goal, midair_init = random_state.choice([True, False], 2)
            if not self.eval_flag and grounded_goal:
                self._initialize_on_table(physics, random_state)
            else:
                self._initialize_midair(physics, random_state)

            self._prepare_goal(physics, random_state)
            physics.bind(self._target_site).pos = self._goal_pos

            if not self.eval_flag and midair_init:
                self._initialize_midair(physics, random_state,
                                        fixed_pos=self._tcp_center)
            else:
                self._initialize_on_table(physics, random_state)

            # Resample successful init.
            if self.get_success(physics):
                self.initialize_episode(physics, random_state)
            # Resample invalid prop placement
            pos, _ = self._prop.get_pose(physics)
            low, high = self._workspace.prop_bbox
            is_invalid = np.logical_or(pos < low, pos > high)
            if np.any(is_invalid[:-1]):
                self.initialize_episode(physics, random_state)
        except PhysicsError:
            self.initialize_episode(physics, random_state)

    def get_success(self, physics):
        """Is the prop properly placed?"""
        pos, _ = self._prop.get_pose(physics)
        dist = np.linalg.norm(pos - self._goal_pos)
        return dist < self._threshold

    # TODO: fix impossible configurations that lead to divergences.
    def _initialize_midair(self, physics, random_state, fixed_pos=None):
        """Prop is somewhere in the air."""
        super().initialize_episode(physics, random_state)
        mocap = physics.bind(self._mocap)
        pos = fixed_pos if fixed_pos is not None else mocap.mocap_pos
        mocap.mocap_pos = pos
        physics.bind(self._hand.actuators[0]).ctrl = 255
        # set_pose only works because prop's parent is a worldbody.
        self._prop.set_pose(physics, pos)
        xvel, qvel = map(
            lambda vel: np.zeros_like(vel.copy()),
            self._prop.get_velocity(physics))
        for _ in range(100):
            self._prop.set_pose(physics, mocap.mocap_pos)
            self._prop.set_velocity(physics, xvel, qvel)
            physics.step()

    def _initialize_on_table(self, physics, random_state):
        """Prop placed on the table"""
        super().initialize_episode(physics, random_state)
        self._prop_placer(physics, random_state)

    def _prepare_goal(self, physics, random_state):
        """Update and add goals to the observation dict."""
        img = self.task_observables['kinect/image']
        self._goal_img = img(physics, random_state).copy()
        pos, _ = self._prop.get_pose(physics)
        self._goal_pos = pos.copy()
