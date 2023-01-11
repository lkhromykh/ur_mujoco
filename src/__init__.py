"""Most of the code can be found in the dm_control repo."""
from dm_control.composer import Environment as Environment

from src.tasks.fetch_pick import FetchPick


_ALL_TASKS = dict(
    fetch=FetchPick
)


class suite:
    """Mimicking dm_control.suite."""
    ALL_TASKS = list(_ALL_TASKS)

    @staticmethod
    def load(name, task_kwargs=None, env_kwargs=None):
        task_kwargs = task_kwargs or {}
        task = _ALL_TASKS[name](**task_kwargs)
        env_kwargs = env_kwargs or {}
        return Environment(
            task,
            raise_exception_on_physics_error=False,
            strip_singleton_obs_buffer_dim=True,
            max_reset_attempts=20,
            **env_kwargs
        )
