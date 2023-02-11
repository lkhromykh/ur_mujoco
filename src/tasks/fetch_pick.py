from typing import Tuple, NamedTuple
from collections import OrderedDict

import numpy as np
from dm_control.composer import initializers
from dm_control.manipulation.shared import workspaces
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions, noises
from dm_control.rl.control import PhysicsError
from dm_control.composer.environment import EpisodeInitializationError

from src import constants
from src.tasks import base
from src.entities.props import primitive

_BOX_MASS = .1
_BOX_SIZE = (.04, .03, .015)
_BOX_OFFSET = np.array([-.5, .05, .1])

_DISTANCE_THRESHOLD = .05
_SCENE_SIZE = .15


class FetchWorkspace(NamedTuple):
    tcp_bbox: workspaces.BoundingBox
    scene_bbox: workspaces.BoundingBox
    arm_offset: np.ndarray
    prop_bbox: workspaces.BoundingBox


_lower = lambda h=0: np.array([-_SCENE_SIZE, -_SCENE_SIZE, h])
_upper = lambda h=0: np.array([_SCENE_SIZE, _SCENE_SIZE, h])

_DEFAULT_WORKSPACE = FetchWorkspace(
    tcp_bbox=workspaces.BoundingBox(
        lower=_BOX_OFFSET + _lower(),
        upper=_BOX_OFFSET + _upper(.3),
    ),
    scene_bbox=workspaces.BoundingBox(
        lower=_BOX_OFFSET + _lower(-_BOX_OFFSET[-1]),
        upper=_BOX_OFFSET + _upper(.3),
    ),
    prop_bbox=workspaces.BoundingBox(
        lower=_BOX_OFFSET + _lower(),
        upper=_BOX_OFFSET + _upper(),
    ),
    arm_offset=constants.ARM_OFFSET
)


class FetchProp(primitive.BoxWithVertexSites):
    """No velocity sensors and touch sensor w/ a cutoff."""

    def _build_observables(self):
        return primitive.StaticPrimitiveObservables(self)


class FetchPick(base.Task):
    """Closely resembles Lift task but require
    to fetch box in a precise position, not just surpass height threshold."""

    def __init__(self,
                 workspace: FetchWorkspace = _DEFAULT_WORKSPACE,
                 control_timestep: float = constants.CONTROL_TIMESTEP,
                 img_size: Tuple[int, int] = (100, 100),
                 distance_threshold: float = _DISTANCE_THRESHOLD,
                 ):
        super().__init__(workspace, control_timestep, img_size)
        self._prop = FetchProp(
            half_lengths=_BOX_SIZE, mass=_BOX_MASS)
        self._prop.geom.rgba = (1, 0, 0, 1)
        self._arena.add_free_entity(self._prop)
        self._threshold = distance_threshold

        tbb = workspace.tcp_bbox
        self._tcp_center = (tbb.lower + tbb.upper) / 2
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand, self._arm,
            position=self._tcp_center
        )
        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*workspace.prop_bbox),
            # quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=False,
            settle_physics=True,
            min_settle_physics_time=1e-2
        )

        self._target_site = workspaces.add_target_site(
            body=self.root_entity.mjcf_model.worldbody,
            radius=distance_threshold,
            visible=False,
            rgba=constants.RED,
            name='target_site'
        )
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower, upper=workspace.prop_bbox.upper,
            rgba=constants.BLUE, name='prop_spawn_area')

        # Duct tape.
        self._task_observables = OrderedDict()
        self.eval_flag = False
        # Spoofing observables so they are included on _build.
        self._goal_pos = np.zeros((3,), np.float64)
        self._task_observables['goal_pos'] = \
            observable.Generic(lambda _: self._goal_pos)

        w, h = img_size
        # in meters
        nearest, farthest = 0.05, 2.
        self._goal_rgbd = np.zeros(img_size + (4,), np.uint8)

        def depth_fn(physics):
            """Mujoco returns distance in meters."""
            depth = physics.render(
                width=w, height=h, camera_id="kinect", depth=True)
            depth = np.where(depth > 6., 0, depth)
            depth *= np.random.random(depth.shape) > .1
            depth += np.random.normal(scale=.02, size=depth.shape)
            depth = (depth - nearest) / (farthest - nearest)
            depth = np.clip(depth, 0, 1)
            return np.uint8(255 * depth)

        def rgbd(physics):
            img = physics.render(width=w, height=h, camera_id="kinect")
            depth = depth_fn(physics)
            return np.concatenate([img, depth[..., np.newaxis]], -1)

        self._task_observables["rgbd"] = observable.Generic(rgbd)
        self._task_observables["goal_rgbd"] = \
            observable.Generic(lambda _: self._goal_rgbd)

        def to_prop(physics):
            pos, _ = self._get_mocap(physics)
            obj_pos, _ = self._prop.get_pose(physics)
            return obj_pos - pos

        self._task_observables['prop_distance'] =\
            observable.Generic(to_prop)

        self.__post_init__()

    def _build_variations(self):
        super()._build_variations()
        self._mjcf_variation.bind_attributes(
            self._prop.geom,
            rgba=noises.Additive(base.RgbVariation(-.3, .3)),
            size=distributions.Uniform(.01, .03),
            mass=distributions.Uniform(.1, .5)
        )

    def _build_observables(self):
        super()._build_observables()
        self._prop.observables.enable_all()

    # def initialize_episode_mjcf(self, random_state):
    #    del random_state

    def initialize_episode(self, physics, random_state):
        try:
            # Goal on the table or in the air.
            #grounded_goal = random_state.choice([True, False], p=[.5, .5])
            grounded_goal = False
            if not self.eval_flag and grounded_goal:
                self._initialize_on_table(physics, random_state)
            else:
                target_pos = random_state.uniform(*self._workspace.tcp_bbox)
                self._initialize_midair(physics, random_state, target_pos)

            self._prepare_goal(physics, random_state)
            physics.bind(self._target_site).pos = self._goal_pos

            # Begin from the grasped state (fixed): this can ease exploration.
            midair_start = random_state.choice([True, False], p=[.5, .5])
            # midair_start = False
            if not self.eval_flag and midair_start:
                self._initialize_midair(
                    physics, random_state, fixed_pos=self._tcp_center)
            else:
                self._initialize_on_table(physics, random_state)
            physics.step(100)
            self._hand.set_grasp(physics, 0.)

            # Resample successful init.
            if self.get_success(physics):
                self.initialize_episode(physics, random_state)
            # Resample invalid due to physics_settle prop placement.
            pos, _ = self._prop.get_pose(physics)
            low, high = self._workspace.prop_bbox
            is_invalid = np.logical_or(pos < low, pos > high)
            if np.any(is_invalid[:-1]):
                self.initialize_episode(physics, random_state)
        except Exception as exp:
            # composer.Environment can handle errors on init.
            raise EpisodeInitializationError(exp) from exp

    def get_success(self, physics):
        """Is the prop properly placed?"""
        pos, _ = self._prop.get_pose(physics)
        dist = np.linalg.norm(pos - self._goal_pos)
        return dist < self._threshold

    def should_terminate_episode(self, physics):
        return False

    # TODO: fix impossible configurations that lead to divergences.
    def _initialize_midair(self, physics, random_state, fixed_pos=None):
        """Prop is somewhere in the air."""
        super().initialize_episode(physics, random_state)
        mocap = physics.bind(self._mocap)
        pos = fixed_pos if fixed_pos is not None else mocap.mocap_pos
        mocap.mocap_pos = pos
        physics.step(100)
        self._hand.set_grasp(physics, 1.)

        # set_pose only works because prop's parent is a worldbody.
        self._prop.set_pose(physics, pos)
        xvel, qvel = map(
            lambda vel: np.zeros_like(vel.copy()),
            self._prop.get_velocity(physics))
        touch = self._prop.observables.touch
        maxiter = 300
        while touch(physics) < .1:
            self._prop.set_pose(physics, mocap.mocap_pos)
            self._prop.set_velocity(physics, xvel, qvel)
            physics.step()
            maxiter -= 1
            if maxiter == 0:
                raise EpisodeInitializationError

    def _initialize_on_table(self, physics, random_state):
        """Prop placed on the table."""
        super().initialize_episode(physics, random_state)
        self._prop_placer(physics, random_state)

    def _prepare_goal(self, physics, random_state):
        """Snap current observations as a desired episode goal."""
        rgbd = self.task_observables["rgbd"]
        self._goal_rgbd = rgbd(physics, random_state).copy()
        pos, _ = self._prop.get_pose(physics)
        self._goal_pos = pos.copy()

    def compute_reward(self, achieved_goal, desired_goal):
        ach_pos = achieved_goal["box/position"]
        des_pos = desired_goal["box/position"]
        dist = np.linalg.norm(ach_pos - des_pos)
        return dist < self._threshold
