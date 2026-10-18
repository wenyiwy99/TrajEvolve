# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        self._seed = seed  # 保存 seed，用于信用分析时精确复现环境
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward
        info['session_id'] = self.env.session  # 保存任务索引，用于信用分析时重新加载相同任务
        info['env_seed'] = self._seed  # 保存 seed，用于信用分析时精确复现环境

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        info['session_id'] = idx  # 保存任务索引，用于信用分析时重新加载相同任务
        info['env_seed'] = self._seed  # 保存 seed，用于信用分析时精确复现环境
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def predict_buy_reward(self):
        """预计算当前 session 执行 click[buy now] 后的 reward"""
        from web_agent_site.engine.goal import get_reward
        
        session_id = self.env.session
        session = self.env.server.user_sessions.get(session_id)
        if not session or not session.get('asin'):
            return 0.0  # 没有选中商品
        
        goal = session['goal']
        asin = session['asin']
        options = session.get('options', {})
        purchased_product = self.env.server.product_item_dict.get(asin)
        price = self.env.server.product_prices.get(asin)
        
        if not purchased_product:
            return 0.0
        
        reward = get_reward(purchased_product, goal, price, options)
        return reward

    def is_product_correct(self):
        """检查当前选中的 ASIN 是否与 goal 的目标 ASIN 匹配"""
        session_id = self.env.session
        session = self.env.server.user_sessions.get(session_id)
        if not session or not session.get('asin'):
            return False
        return session['asin'].upper() == session['goal']['asin'].upper()

    def get_attribute_options(self):
        """获取当前可用的属性选项（过滤掉导航按钮）"""
        available = self.env.get_available_actions()
        if not available or 'clickables' not in available:
            return []
        
        # 过滤掉非属性选项
        nav_buttons = {'back to search', '< prev', 'next >', 'description', 
                       'features', 'reviews', 'buy now'}
        options = [c for c in available['clickables'] 
                   if c.lower() not in nav_buttons and not c.lower().startswith('b0')]  # b0 是 ASIN 前缀
        return options

    def get_first_missing_goal_option(self):
        """获取第一个未完成的 goal 选项值"""
        session_id = self.env.session
        session = self.env.server.user_sessions.get(session_id)
        if not session:
            return None
        
        goal_options = session['goal'].get('goal_options', {})
        current_options = session.get('options', {})
        
        # goal_options 格式: {'color': 'multi 9', 'size': 'medium'}
        # current_options 格式: {'color': 'xxx', 'size': 'yyy'} (已选择的值)
        for key, goal_value in goal_options.items():
            current_value = current_options.get(key)
            if current_value is None or current_value.lower() != goal_value.lower():
                # 这个选项还未正确选择，返回 goal 要求的值
                return goal_value
        
        return None  # 所有选项都已正确完成

    def get_goal_asin(self):
        """获取当前任务的目标商品 ASIN"""
        session_id = self.env.session
        session = self.env.server.user_sessions.get(session_id)
        if not session or 'goal' not in session:
            return None
        return session['goal'].get('asin', '').lower()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)
        
        # Store previous available_actions for admissible mapping (similar to ALFWorld's prev_admissible_commands)
        self.prev_available_actions = [None] * self.num_processes

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, (obs, reward, done, info) in enumerate(results):
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            # Update prev_available_actions from info
            self.prev_available_actions[i] = info.get('available_actions', None)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for i, (obs, info) in enumerate(results):
            obs_list.append(obs)
            info_list.append(info)
            # Update prev_available_actions from info
            self.prev_available_actions[i] = info.get('available_actions', None)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)
    
    @property
    def get_available_actions(self):
        """
        Return the prev_available_actions stored from last step/reset.
        Similar to ALFWorld's get_admissible_commands property.
        """
        return self.prev_available_actions
    
    def predict_buy_rewards(self):
        """批量获取所有环境的预计算购买奖励"""
        futures = [worker.predict_buy_reward.remote() for worker in self._workers]
        return ray.get(futures)

    def is_products_correct(self):
        """批量检查所有环境的产品是否正确"""
        futures = [worker.is_product_correct.remote() for worker in self._workers]
        return ray.get(futures)

    def get_all_attribute_options(self):
        """批量获取所有环境的属性选项"""
        futures = [worker.get_attribute_options.remote() for worker in self._workers]
        return ray.get(futures)

    def get_all_first_missing_goal_options(self):
        """批量获取所有环境的第一个未完成 goal 选项"""
        futures = [worker.get_first_missing_goal_option.remote() for worker in self._workers]
        return ray.get(futures)

    def get_goal_asins(self):
        """批量获取所有环境的目标 ASIN"""
        futures = [worker.get_goal_asin.remote() for worker in self._workers]
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )


# -----------------------------------------------------------------------------
# Evolution Environment (seed + goal_idx based, for evolution trajectory collection)
# -----------------------------------------------------------------------------

class WebshopEvolveEnvs:
    """
    Manages multiple WebShop environments for evolution trajectory collection.
    
    Each trajectory is precisely reconstructed using (env_seed, goal_idx) pairs.
    Internally uses WebshopWorker Ray actors (same as WebshopMultiProcessEnv).
    
    Interface is compatible with WebshopMultiProcessEnv so it can be wrapped by
    WebshopEnvironmentManager.
    """
    
    def __init__(self, seed_goal_pairs, resources_per_worker=None):
        """
        Args:
            seed_goal_pairs: List of (env_seed, goal_idx) tuples, one per trajectory
            resources_per_worker: Ray resource allocation per worker (optional)
        """
        if not ray.is_initialized():
            ray.init()
        
        self.seed_goal_pairs = seed_goal_pairs
        self.num_processes = len(seed_goal_pairs)
        
        if resources_per_worker is None:
            resources_per_worker = {"num_cpus": 0.5}
        
        env_kwargs = {
            'observation_mode': 'text',
            'human_goals': False,
            'num_prev_obs': 0,
            'num_prev_actions': 0,
        }
        
        # Create WebshopWorker Ray actors, each with its own seed
        WorkerActor = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for env_seed, _ in seed_goal_pairs:
            worker = WorkerActor.remote(seed=env_seed, env_kwargs=env_kwargs)
            self._workers.append(worker)
        
        self.prev_available_actions = [None] * self.num_processes
    
    def step(self, actions):
        """Execute actions in all environments in parallel."""
        assert len(actions) == self.num_processes, \
            f"Number of actions ({len(actions)}) must equal number of environments ({self.num_processes})"
        
        futures = [worker.step.remote(action) for worker, action in zip(self._workers, actions)]
        results = ray.get(futures)
        
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, (obs, reward, done, info) in enumerate(results):
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self.prev_available_actions[i] = info.get('available_actions', None)
        
        return obs_list, reward_list, done_list, info_list
    
    def reset(self):
        """Reset all environments to their respective goal_idx in parallel."""
        futures = [
            worker.reset.remote(goal_idx)
            for worker, (_, goal_idx) in zip(self._workers, self.seed_goal_pairs)
        ]
        results = ray.get(futures)
        
        obs_list, info_list = [], []
        for i, (obs, info) in enumerate(results):
            obs_list.append(obs)
            info_list.append(info)
            self.prev_available_actions[i] = info.get('available_actions', None)
        
        return obs_list, info_list
    
    @property
    def get_available_actions(self):
        """Return the prev_available_actions stored from last step/reset."""
        return self.prev_available_actions
    
    def close(self):
        """Close all workers."""
        if getattr(self, '_closed', False):
            return
        
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        ray.get(close_futures)
        
        for worker in self._workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        
        self._closed = True
    
    def __del__(self):
        self.close()


def build_evolution_envs_webshop(seed_goal_pairs, config, resources_per_worker=None):
    """
    Create an independent environment manager for evolution trajectory collection.
    
    Uses (env_seed, goal_idx) pairs to precisely reconstruct each trajectory's
    environment. This creates a completely separate set of environments that won't
    interfere with the main training loop's environments.
    
    Args:
        seed_goal_pairs: List of (env_seed, goal_idx) tuples
        config: Configuration object (used by WebshopEnvironmentManager)
        resources_per_worker: Ray resource allocation per worker (optional)
        
    Returns:
        WebshopEnvironmentManager instance wrapping WebshopEvolveEnvs
        
    Example:
        >>> seed_goal_pairs = [(42, 501), (42, 502), (7, 600)]
        >>> envs = build_evolution_envs_webshop(seed_goal_pairs, config)
        >>> obs, infos = envs.reset(kwargs=None)
        >>> next_obs, rewards, dones, infos = envs.step(actions)
        >>> envs.close()
    """
    from functools import partial
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import webshop_projection
    
    _envs = WebshopEvolveEnvs(seed_goal_pairs, resources_per_worker)
    projection_f = partial(webshop_projection)
    return WebshopEnvironmentManager(_envs, projection_f, config)