import abc

import numpy as np
from dm_env import specs
from dm_control import composer
from dm_control.composer import observation
from dm_control.composer import initializers
from dm_control.manipulation.shared import workspaces
from dm_control.composer.variation import distributions

from src import consts
from src.entities import Arena, Robotiq2f85, UR5e, primitive, cameras


class BaseTask(composer.Task):
    def __init__(self, workspace, control_timestep):
        self._arena = Arena()
        self._arm = UR5e()
        self._hand = Robotiq2f85()
        _attach_hand_to_arm(self._arm, self._hand)
        self._arena.attach_offset(self._arm, workspace.arm_offset)

        self.control_timestep = control_timestep
        self._task_observables = cameras.add_camera_observables(
            self._arena, cameras.FRONT_CLOSE, height=64, width=64
        )
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand, self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox),
            quaternion=np.array([1, 0, 0, 0.]),
        )

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode(self, physics, random_state):
        # self._hand.set_grasp(physics, 0)
        physics.bind(self._arm.joints).qpos = consts.HOME
        self._tcp_initializer(physics, random_state)

    @property
    def arm(self):
        return self._arm

    @property
    def hand(self):
        return self._hand


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
