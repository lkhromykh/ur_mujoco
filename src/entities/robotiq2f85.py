import os

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.entities.manipulators.base import RobotHand

from src import constants

_ROBOTIQ2F85_XML_PATH = os.path.join(
    os.path.dirname(__file__),
    '../third_party/mujoco_menagerie/robotiq_2f85/2f85.xml'
)


class Robotiq2f85(RobotHand):
    def _build(self):
        self._mjcf_model = mjcf.from_path(_ROBOTIQ2F85_XML_PATH)
        self._actuators = self.mjcf_model.find_all('actuator')
        self._joints = self.mjcf_model.find_all('joint')
        self._base = self.mjcf_model.find('body', 'base_mount')
        self._bodies = self.mjcf_model.find_all('body')
        self._tcp_site = self._base.add(
            'site',
            name='tcp_center_point',
            pos=[0, 0, .1493],
            group=constants.TASK_SITE_GROUP
        )

    def _build_observables(self):
        return RobotiqObservables(self)

    def set_grasp(self, physics, close_factor):
        """[0., 1.] -> uint8"""
        ctrl = int(255 * close_factor)
        physics.set_control(ctrl)

    @property
    def tool_center_point(self):
        return self._tcp_site

    @property
    def actuators(self):
        return self._actuators

    @property
    def mjcf_model(self):
        return self._mjcf_model

    @property
    def joints(self):
        return self._joints

    @property
    def bodies(self):
        return self._bodies

    @property
    def base(self):
        return self._base


class RobotiqObservables(composer.Observables):

    @composer.observable
    def tcp_pose(self):
        return observable.MJCFFeature('xpos', self._entity.tool_center_point)

    @composer.observable
    def tcp_rmat(self):
        return observable.MJCFFeature('xmat', self._entity.tool_center_point)
