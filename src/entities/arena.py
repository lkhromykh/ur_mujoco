import os

from dm_control import mjcf
from dm_control.composer import Entity

_ARENA_XML_PATH = os.path.join(os.path.dirname(__file__), 'arena.xml')


class Arena(Entity):
    def _build(self):
        self._mjcf_root = mjcf.from_path(_ARENA_XML_PATH)
        self.mjcf_model.worldbody.add(
            'geom',
            name='ground',
            type='plane',
            material='groundplane',
            size=(2, 2, 0.1),
            friction=(.4,),
            solimp=(.95, .99, .001),
            solref=(.002, 1.)
        )
        # self.mjcf_model.worldbody.add(
        #     'light',
        #     pos=(0, 0, 1.5),
        #     dir=(0, 0, -1),
        #     diffuse=(.7, .7, .7),
        #     specular=(.3, .3, .3),
        #     directional='false',
        #     castshadow='true'
        # )
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


def _connect_mocap(body):
    mocap = mjcf.RootElement()
    pos = (0, 0, 0)
    mc_body = mocap.worldbody.add(
        'body',
        name='mocap',
        pos=pos,
        mocap='true'
    )
    mc_body.add(
        'geom',
        name='mocap',
        pos=pos,
        type='sphere',
        conaffinity=0,
        contype=0,
        size='.03'
    )
    mc_body.add(
        'site',
        name='mocap',
        pos=pos,
        type='box',
        size='.03'
    )
    mocap.equality.add(
        'weld',
        name='mocap_weld',
        body1=mc_body,
        body2=body,
        solref="0.02 1"
    )
    return mocap
