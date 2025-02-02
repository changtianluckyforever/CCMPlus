import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Dict
from scipy import stats

from env.models_info import ModelSelector
from utils.MMsK_queue import MMSKQueue


def get_mmsk_metrics(req_num, bsz, max_bsz, pod_num, model_latency, timeout, interval=30 * 60):
    arrival_rate = req_num / interval * timeout
    pr_poisson = stats.poisson.cdf(arrival_rate, bsz)
    base_latency = (1 - pr_poisson) * timeout + pr_poisson * (interval * bsz) / req_num
    # todo check
    batch_num = min(req_num / bsz, interval / base_latency)
    lamb = batch_num / interval
    mu = 1 / model_latency
    service_level = 0.9
    service_level_timout = timeout
    # buf_size = int((service_level_timout / model_latency - 1) * pod_num)
    buf_size = int(service_level_timout / model_latency)
    k = buf_size + pod_num
    mmsk_dict = MMSKQueue(lamb, mu, pod_num, k, service_level_timout, service_level).metrics()
    mmsk_dict.update({
        "sys_util": mmsk_dict["util"] * bsz / max_bsz,
        "sys_lat": base_latency + mmsk_dict["w"],
        "violate": mmsk_dict["pr_balk"],
        "req_num": req_num,
        "lamb": lamb,
        "bsz": bsz,
        "max_bsz": max_bsz,
        "parallelism": pod_num,
        "buf_size": buf_size,
        "model_latency": model_latency,
        "timeout": timeout,
    })
    return mmsk_dict


class ModelServingQueueEnv(gym.Env):
    def __init__(self, args, **kwargs):
        self.rank = kwargs.get("rank", 0)
        # logger.info("-" * 20 + f" init {self.rank}-th Env " + "-" * 20)
        self.args = args.env_args
        self.num_models = self.args.num_models
        self.window_len = self.args.window_day * 24 * 2
        self.time_limit = (self.args.days_limit + self.args.window_day) * 24 * 2
        self.model_selector = ModelSelector(k=self.num_models)
        self.to_csv = self.args.to_csv

        self.observation_space = Dict(
            {"cur_obs": Box(low=0., high=np.inf, shape=(self.num_models, args.env_args.feat_dim)),
             "seq_obs": Box(low=0., high=np.inf, shape=(self.num_models, self.window_len, args.env_args.seq_feat_dim))}
        )
        self.action_space = Box(low=-1., high=1., shape=(self.num_models, 2))

    def _init_models(self):
        self.model_selector.sample_models()
        self.model_gpus, self.model_cpus, self.model_mems, self.model_max_bsz, self.model_bsz, self.model_pal, \
            self.model_lat, self.model_max_lat, self.model_launch_cost = self.model_selector.models_info()
        self.model_req, self.ts_feat = self.model_selector.requests(trace_days=self.args.days_limit,
                                                                    init_days=self.args.window_day)
        self.obs_log = None
        self.cur_step = self.window_len

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.free_gpus = self.args.gpu_per_node * self.args.num_node
        self.free_cpus = self.args.cpu_per_node * self.args.num_node
        self.free_mems = self.args.mem_per_node * self.args.num_node
        self._init_models()
        self.step(np.array([self.model_bsz / self.model_max_bsz, np.zeros(self.num_models)]).transpose())
        return {"cur_obs": self._get_cur_obs().to_numpy(), "seq_obs": self._get_seq_obs()}, {}

    def action_map(self, action: np.array):
        new_model_bsz, new_model_pal = action[:, 0], action[:, 1]
        new_model_bsz = ((new_model_bsz + 1) / 2 * self.model_max_bsz).astype(int)
        # todo max parallelism limit
        new_model_pal = np.maximum(((1 + new_model_pal) * self.model_pal).astype(int), 1)
        return new_model_bsz, new_model_pal

    def step(self, action: np.array):
        new_model_bsz, new_model_pal = self.action_map(action)
        # print("action: ", new_model_bsz, new_model_pal)
        self.model_bsz = new_model_bsz
        allocate_pod_penalty = 0
        old_model_pal = self.model_pal
        # todo clip by ratio
        mis = np.arange(self.num_models)
        np.random.shuffle(mis)
        # for mi, (new_p, old_p) in enumerate(zip(new_model_pal, old_model_pal)):
        for mi in mis:
            new_p, old_p = new_model_pal[mi], old_model_pal[mi]
            if new_p > old_p:
                while self.free_gpus < (new_p - old_p) * self.model_gpus[mi][new_model_bsz[mi]]:
                    new_p -= 1
                    allocate_pod_penalty += 1
                while self.free_cpus < (new_p - old_p) * self.model_cpus[mi][new_model_bsz[mi]]:
                    new_p -= 1
                    allocate_pod_penalty += 1
                while self.free_mems < (new_p - old_p) * self.model_mems[mi][new_model_bsz[mi]]:
                    new_p -= 1
                    allocate_pod_penalty += 1
                new_model_pal[mi] = new_p
            # assume that all pods can be started and destroyed before next action step
            self.free_gpus += (old_p - new_p) * self.model_gpus[mi][new_model_bsz[mi]]
            self.free_cpus += (old_p - new_p) * self.model_cpus[mi][new_model_bsz[mi]]
            self.free_mems += (old_p - new_p) * self.model_mems[mi][new_model_bsz[mi]]
            self.model_pal[mi] = new_p
        obs = self._get_cur_obs()
        reward = obs["sys_util"].mean() - obs["sys_lat"].mean() - obs["violate"].mean() - allocate_pod_penalty
        self.cur_step += 1
        terminated = self.cur_step == self.time_limit - 1
        if terminated:
            if self.to_csv:
                self.obs_log.to_csv(f"obs_log_{self.rank}.csv")
        info = {}
        truncated = False
        return {"cur_obs": obs.to_numpy(), "seq_obs": self._get_seq_obs()}, reward, terminated, truncated, info

    def _get_cur_obs(self):
        service_obs = []
        cluster_obs = {"free_cpu": self.free_cpus, "free_gpu": self.free_gpus, "free_mem": self.free_mems}
        for mi in range(self.num_models):
            m_dict = get_mmsk_metrics(self.model_req[mi, self.cur_step], np.exp2(self.model_bsz[mi]).astype(int),
                                      self.model_max_bsz[mi], self.model_pal[mi],
                                      self.model_lat[mi, self.model_bsz[mi]], self.model_max_lat[mi])
            m_dict.update(cluster_obs)
            m_dict["mid"] = f"M{mi}"
            service_obs.append(m_dict)
        service_obs = pd.DataFrame(service_obs)
        if self.to_csv:
            if self.obs_log is None:
                self.obs_log = service_obs
            else:
                self.obs_log = pd.concat([self.obs_log, service_obs])
        return service_obs.set_index("mid")

    def _get_seq_obs(self):
        s, e = self.cur_step - self.window_len + 1, self.cur_step + 1
        return np.concatenate([self.model_req[:, s:e], self.ts_feat[:, s:e]], axis=0)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
