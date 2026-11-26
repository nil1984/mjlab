from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfgs import unitree_g1_flat_env_cfg
from .rl_cfg import unitree_g1_ppo_runner_cfg

def g1_soccer_walk_env_cfg(play: bool=False):
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.commands["twist"].rel_standing_envs = 0.2
  cfg.rewards["foot_slip"].weight = -0.5
  cfg.rewards["air_time"].weight = 0.5
  return cfg

def g1_soccer_walk_rl_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_soccer_walk"
  cfg.max_iterations = 5000
  return cfg

register_mjlab_task(
    task_id = "Mjlab-SoccerWalk-Flat-Unitree-G1",
    env_cfg = g1_soccer_walk_env_cfg,
    play_env_cfg= g1_soccer_walk_env_cfg(play=True),
    rl_cfg = g1_soccer_walk_rl_cfg,
    runner_cls = VelocityOnPolicyRunner,
)
