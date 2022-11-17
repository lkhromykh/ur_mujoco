import os

from dm_control import mjcf
from dm_control.composer import Entity

from src import constants

_ARENA_XML_PATH = os.path.join(os.path.dirname(__file__), 'arena.xml')


class Arena(Entity):

    def _build(self):
        self._mjcf_root = mjcf.from_path(_ARENA_XML_PATH)

        self._groundplane = self.mjcf_model.asset.add(
            'material',
            name='groundplane',
            specular=.7,
            shininess=.1,
            reflectance=.01,
            rgba=(.4, .4, .4, 1.)
        )
        self._ground = self.mjcf_model.worldbody.add(
            'geom',
            name='ground',
            type='plane',
            material=self._groundplane,
            size=(4, 1.5, 0.01),
            friction=(.4,),
            solimp=(.95, .99, .001),
            solref=(.002, 1.)
        )
        self.mjcf_model.worldbody.add(
            'light',
            name='lab',
            pos=(0, 0, 2.),
            dir=(0, 0, -1),
            diffuse=(.6, .6, .6),
            specular=(.3, .3, .3),
            ambient=(.4, .4, .4),
            directional='true',
            castshadow='false'
        )
        self.mjcf_model.worldbody.add(
            'light',
            name='hall',
            pos=(0, -2., 1.),
            dir=(0, 1, 0),
            diffuse=(.2, .2, .2),
            specular=(.2, .2, .2),
            directional='true',
            castshadow='false'
        )
        self.mjcf_model.statistic.center = (0., 0., 0.)

    def add_free_entity(self, entity):
        frame = self.attach(entity)
        frame.add('freejoint')
        return frame

    def attach_offset(self, entity, offset, attach_site=None):
        frame = self.attach(entity, attach_site=attach_site)
        frame.pos = offset
        return frame

    def insert_mocap(self, body: mjcf.Element):
        mocap = _connect_mocap(body)
        self.mjcf_model.include_copy(mocap)
        return self.mjcf_model.find('body', 'mocap')

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def groundplane(self):
        return self._groundplane


def _connect_mocap(body: mjcf.Element):
    mocap = mjcf.RootElement()
    pos = (0, 0, 0)
    mc_body = mocap.worldbody.add(
        'body',
        name='mocap',
        pos=pos,
        mocap='true',
    )
    # mc_body.add(
    #     'geom',
    #     name='mocap',
    #     pos=pos,
    #     type='sphere',
    #     conaffinity=0,
    #     contype=0,
    #     size='.03'
    # )
    mc_body.add(
        'site',
        name='mocap',
        pos=pos,
        type='sphere',
        size='.03',
        group=constants.TASK_SITE_GROUP
    )
    mocap.equality.add(
        'weld',
        name='mocap_weld',
        body1=mc_body,
        body2=body,
        solref="0.02 1"
    )
    return mocap
