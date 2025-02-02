import random
from typing import List

import gym
import numpy as np
from gym.spaces import Box

from env.models_info import ModelSelector

SEQ_FEAT_DIM = 7


class ModelServingEnv(gym.Env):
    def __init__(self, args, **kwargs):
        self.rank = kwargs.get("rank", 0)
        self.seed(args.seed + self.rank)
        # logger.info("-" * 20 + f" init {self.rank}-th Env " + "-" * 20)
        self.args = args.env_args
        self.num_models = self.args.num_models
        self.time_limit = self.args.episode_limit
        self.step_interval = self.args.step_interval
        self.model_selector = ModelSelector(k=self.num_models)

        self.observation_space = Box(low=0., high=np.inf, shape=(self.num_models, self.step_interval, SEQ_FEAT_DIM))
        self.action_space = Box(low=-1., high=1., shape=(self.num_models, 2))

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def _init_models(self):
        self.model_selector.sample_models()
        self.model_gpus, self.model_cpus, self.model_mems, self.model_max_bsz, self.model_bsz, self.model_pal, \
            self.model_lat, self.model_max_lat, self.model_launch_cost = self.model_selector.models_info()
        self.model_req, self.req_sizes = self.model_selector.requests()
        # self.req_latency = np.zeros_like(self.model_req)
        self.model_batch_t = np.zeros(self.num_models)
        self.cur_time = 0.
        self.cur_req_idx = np.zeros(self.num_models, dtype=np.int)
        self.pod_avail_time = [[0. for _ in range(self.model_pal[i])] for i in range(self.num_models)]
        self.pod_free_duration = []
        self.model_finish = np.zeros(self.num_models)
        self.model_util = np.zeros(self.num_models)
        self.latency_stat = []
        self.latency_violate = 0
        self.processed_batch_num = np.zeros(self.num_models, dtype=np.int)

    def reset(self):
        self.free_gpus = self.args.gpu_per_node * self.args.num_node
        self.free_cpus = self.args.cpu_per_node * self.args.num_node
        self.free_mems = self.args.mem_per_node * self.args.num_node
        self._init_models()
        self.step([self.model_bsz, self.model_pal])
        return self._get_obs()

    def step(self, action: List):
        # [batch_num, processed_req_num, total_req_num, avg_latency, latency_violate, avg_free_time, avg_util]
        self.step_stat = np.zeros((self.num_models, self.step_interval, SEQ_FEAT_DIM))
        new_model_bsz, new_model_pal = action[0], action[1]
        self.model_bsz = np.exp2(new_model_bsz).astype(int)
        self.model_util = new_model_bsz / self.model_max_bsz
        reward_throughput = self.model_util.mean()
        allocate_pod_penalty = 0
        reward_latency, reward_latency_violate = [], []
        new_pod_start_time = []
        old_model_pal = self.model_pal
        mis = np.arange(self.num_models)
        np.random.shuffle(mis)
        # for mi, (new_p, old_p) in enumerate(zip(new_model_pal, old_model_pal)):
        for mi in mis:
            new_p, old_p = new_model_pal[mi], old_model_pal[mi]
            if new_p <= old_p:
                self.pod_avail_time[mi] = self.pod_avail_time[mi][: new_p]
                new_pod_start_time.append(-1)
            else:
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
                new_pod_start_time.append(self.cur_time + self.model_launch_cost[mi])
            # assume that all pods can be started and destroyed before next action step
            self.free_gpus += (old_p - new_p) * self.model_gpus[mi][new_model_bsz[mi]]
            self.free_cpus += (old_p - new_p) * self.model_cpus[mi][new_model_bsz[mi]]
            self.free_mems += (old_p - new_p) * self.model_mems[mi][new_model_bsz[mi]]
            self.model_pal[mi] = new_p

        def next_batch(m_idx):
            prev_idx = self.cur_req_idx[m_idx]
            batch_end = min(self.req_sizes[m_idx], prev_idx + self.model_bsz[m_idx])
            # timeout constrain: collect batch before timeout limit, i.e., max latency
            # batch_end should be at least prev_idx, i.e., batch_size >= 1
            while batch_end > prev_idx + 1 and self.model_req[m_idx][batch_end - 1] - self.model_req[m_idx][prev_idx] > \
                    self.model_max_lat[m_idx]:
                batch_end -= 1
            if batch_end - prev_idx < self.model_bsz[m_idx]:
                batch_end_t = self.model_batch_t[m_idx] + self.model_max_lat[m_idx]
            else:
                batch_end_t = self.model_req[m_idx][batch_end - 1]
            return batch_end, batch_end_t

        for mi in range(self.num_models):
            # try to collect a batch for mode m_i under the timeout constrain
            next_batch_end, batch_collected_t = next_batch(mi)
            while next_batch_end <= self.req_sizes[mi] and batch_collected_t <= self.cur_time + self.step_interval:
                if self.model_finish[mi]:
                    break
                if next_batch_end == self.req_sizes[mi]:
                    self.model_finish[mi] = 1
                # start new added pods before the batch is executed
                if 0 < new_pod_start_time[mi] < batch_collected_t:
                    self.pod_avail_time[mi].extend(
                        [new_pod_start_time[mi] for _ in range(new_model_pal[mi] - old_model_pal[mi])])
                    new_pod_start_time[mi] = -1  # only execute once
                self.pod_avail_time[mi].sort()
                batch_finish_time = self.model_lat[mi][new_model_bsz[mi]] + max(self.pod_avail_time[mi][0],
                                                                                batch_collected_t)
                if batch_finish_time > self.cur_time + self.step_interval:
                    break
                # this collected batch can be executed before next action, update m_i's current time
                self.model_batch_t[mi] = batch_collected_t
                self.processed_batch_num[mi] += 1
                self.pod_avail_time[mi].pop(0)
                self.pod_avail_time[mi].append(batch_finish_time)
                batch_latency = batch_finish_time - self.model_req[mi][self.cur_req_idx[mi]: next_batch_end]
                # self.req_latency[mi, self.cur_req_idx[mi]: next_batch_end] = batch_latency
                reward_latency.append(batch_latency.mean())
                self.latency_stat.append(batch_latency.mean())
                # reward_latency_violate.append((batch_latency > self.model_max_lat[mi]).sum())
                self.cur_req_idx[mi] = next_batch_end
                next_batch_end, batch_collected_t = next_batch(mi)
        reward = reward_throughput - allocate_pod_penalty
        reward -= np.mean(reward_latency) if len(reward_latency) > 0 else np.mean(self.latency_stat)
        self.cur_time += self.step_interval
        done = self.cur_time > self.time_limit or self.model_finish.sum() == self.num_models
        info = {}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        win_batch_num = None
        win_latency = None
        win_violate = None
        model_state = np.stack([self.model_util, self.model_pal, win_batch_num, win_violate, win_latency])
        workload_state = None
        cluster_state = [self.free_gpus, self.free_cpus, self.free_mems]

    def render(self, mode="human"):
        pass

    def close(self):
        pass
