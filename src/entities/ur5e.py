import os

from dm_control import mjcf
from dm_control.entities.manipulators.base import RobotArm

_UR5E_XML_PATH = os.path.join(
    os.path.dirname(__file__),
    '../third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml'
)


class UR5e(RobotArm):
    def _build(self):
        self._mjcf_model = mjcf.from_path(_UR5E_XML_PATH)
        self._mjcf_model.find('light', 'spotlight').remove()
        self._mjcf_model.keyframe.remove()
        self._joints = self.mjcf_model.find_all('joint')
        self._wrist_site = self.mjcf_model.find('site', 'attachment_site')
        self._actuators = self.mjcf_model.find_all('actuator')
        self._bodies = self.mjcf_model.find_all('body')

    @property
    def joints(self):
        return self._joints

    @property
    def wrist_site(self):
        return self._wrist_site

    @property
    def actuators(self):
        return self._actuators

    @property
    def mjcf_model(self):
        return self._mjcf_model
