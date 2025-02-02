import numpy as np
from env.msq_env import ModelServingQueueEnv
from utils.config_util import config_parser
from utils.helper_util import time_block


def hpa(old_p, metric, desired_metric):
    return np.minimum(np.maximum(metric / desired_metric, -1.), 1.)
    # return np.minimum(1, old_p * np.maximum(metric / desired_metric, 2).astype(int))


def run_heuristic(args):
    env = ModelServingQueueEnv(args)
    # envs = AsyncVectorEnv([lambda rank=tr: ModelServingEnv(args, rank=rank) for tr in range(args.n_rollout_threads)])
    done = False
    with time_block("run_heuristic"):
        obs, info = env.reset()
        while not done:
            if args.algo == "hpa":
                new_pal = hpa(obs[:, 18], obs[:, 18], env.model_selector.desired_utils)
            act = np.array([env.model_bsz / env.model_max_bsz, new_pal]).transpose()
            obs, reward, terminated, truncated, info = env.step(act)
            # print(reward)


env_config = config_parser("config/envs/ms.yaml")
env_config.algo = "hpa"
run_heuristic(env_config)
