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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)


class SingleGameEnvWrapper:
    """
    Wrapper for loading a single gamefile as an independent environment.
    
    Reuses AlfredTWEnv's Demangler and wrapper logic to ensure correct observation format.
    This environment is completely independent and does not affect other environments' loading order.
    """
    
    def __init__(self, gamefile: str, alf_config_path: str = None):
        """
        Args:
            gamefile: Game file path (e.g., .../game.tw-pddl)
            alf_config_path: ALFWorld config file path (optional, defaults to config_tw.yaml)
        """
        import textworld
        import textworld.gym
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler, AlfredInfos
        )
        
        self.gamefile = gamefile
        self.env = None
        
        # Use default config path
        if alf_config_path is None:
            alf_config_path = os.path.join(
                os.path.dirname(__file__), 
                "configs", 
                "config_tw.yaml"
            )
        
        # Load config
        self.config = load_config_file(alf_config_path)
        
        # Create wrappers (same logic as AlfredTWEnv.init_env)
        alfred_demangler = AlfredDemangler(shuffle=False)  # No randomization
        wrappers = [alfred_demangler, AlfredInfos]
        
        # Set requested info
        request_infos = textworld.EnvInfos(
            won=True, 
            admissible_commands=True, 
            extras=["gamefile"]
        )
        
        # Register environment with only this gamefile
        self._env_id = textworld.gym.register_games(
            [self.gamefile],
            request_infos,
            batch_size=1,
            asynchronous=False,  # Synchronous execution, more lightweight
            max_episode_steps=50,
            wrappers=wrappers
        )
        
        # Create environment
        self.env = textworld.gym.make(self._env_id)
    
    def reset(self):
        """Reset environment to initial state"""
        obs, info = self.env.reset()
        # Unwrap info values from list format (textworld returns batch format)
        for k in list(info.keys()):
            if isinstance(info[k], list) and len(info[k]) == 1:
                info[k] = info[k][0]
        # Return format consistent with AlfworldWorker
        return obs[0], info
    
    def step(self, action: str):
        """Execute an action"""
        obs, scores, dones, info = self.env.step([action])
        # Unwrap info values from list format (textworld returns batch format)
        for k in list(info.keys()):
            if isinstance(info[k], list) and len(info[k]) == 1:
                info[k] = info[k][0]
        # Return format: obs, score, done, info
        return obs[0], scores[0], dones[0], info
    
    def close(self):
        """Close environment"""
        if self.env is not None:
            self.env.close()
            self.env = None


def create_single_game_env(gamefile: str, alf_config_path: str = None) -> SingleGameEnvWrapper:
    """
    Create an independent environment that loads only the specified gamefile.
    
    Reuses AlfredTWEnv's loading logic (including Demangler) to ensure correct observation format.
    This environment is completely independent and does not affect other environments' loading order.
    
    Args:
        gamefile: Game file path (e.g., .../game.tw-pddl)
        alf_config_path: ALFWorld config file path (optional)
        
    Returns:
        SingleGameEnvWrapper instance
        
    Example:
        >>> env = create_single_game_env("/path/to/game.tw-pddl")
        >>> obs, info = env.reset()
        >>> obs, score, done, info = env.step("go to drawer 1")
        >>> env.close()
    """
    return SingleGameEnvWrapper(gamefile, alf_config_path)


class GamefileBasedWorker:
    """
    Ray Actor that loads a specific gamefile instead of sampling from base_env.
    
    Unlike AlfworldWorker which samples gamefiles from a dataset, this worker
    loads a specific gamefile provided at initialization. Used for evolution
    trajectory collection where we need to replay specific game scenarios.
    """
    
    def __init__(self, gamefile: str, alf_config_path: str = None):
        """
        Args:
            gamefile: Game file path (e.g., .../game.tw-pddl)
            alf_config_path: ALFWorld config file path (optional)
        """
        import textworld
        import textworld.gym
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler, AlfredInfos
        )
        
        self.gamefile = gamefile
        self.env = None
        
        if not gamefile:
            # Empty gamefile - create a dummy state
            self._is_valid = False
            return
        
        self._is_valid = True
        
        # Use default config path
        if alf_config_path is None:
            alf_config_path = os.path.join(
                os.path.dirname(__file__), 
                "configs", 
                "config_tw.yaml"
            )
        
        # Load config
        self.config = load_config_file(alf_config_path)
        
        # Create wrappers (same logic as AlfredTWEnv.init_env)
        alfred_demangler = AlfredDemangler(shuffle=False)  # No randomization
        wrappers = [alfred_demangler, AlfredInfos]
        
        # Set requested info
        request_infos = textworld.EnvInfos(
            won=True, 
            admissible_commands=True, 
            extras=["gamefile"]
        )
        
        # Register environment with only this gamefile
        self._env_id = textworld.gym.register_games(
            [self.gamefile],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=50,
            wrappers=wrappers
        )
        
        # Create environment
        self.env = textworld.gym.make(self._env_id)
    
    def step(self, action: str):
        """Execute a step in the environment"""
        if not self._is_valid:
            return [''], [0.0], [True], {'won': False, 'admissible_commands': [], 'is_action_valid': False}
        
        obs, scores, dones, info = self.env.step([action])
        # Format info to match AlfworldWorker output
        for k in list(info.keys()):
            if isinstance(info[k], list) and len(info[k]) == 1:
                info[k] = info[k][0]
        info['observation_text'] = obs
        return obs, scores, dones, info
    
    def reset(self):
        """Reset the environment"""
        if not self._is_valid:
            return [''], {'won': False, 'admissible_commands': [], 'is_action_valid': False}
        
        obs, info = self.env.reset()
        # Format info to match AlfworldWorker output
        for k in list(info.keys()):
            if isinstance(info[k], list) and len(info[k]) == 1:
                info[k] = info[k][0]
        info['observation_text'] = obs
        return obs, info
    
    def close(self):
        """Close environment"""
        if self.env is not None:
            self.env.close()
            self.env = None


class GamefileBasedAlfworldEnvs:
    """
    Manages multiple environments based on specified gamefiles.
    
    Unlike AlfworldEnvs which samples gamefiles from a training/eval dataset,
    this class creates environments for specific gamefiles. Used for evolution
    trajectory collection.
    
    Interface is compatible with AlfworldEnvs so it can be wrapped by
    AlfWorldEnvironmentManager.
    """
    
    def __init__(self, gamefiles, resources_per_worker=None):
        """
        Args:
            gamefiles: List of game file paths
            resources_per_worker: Ray resource allocation per worker (optional)
        """
        from typing import List
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.gamefiles = gamefiles
        self.num_processes = len(gamefiles)
        self.multi_modal = False  # Text-only for evolution
        
        # Default resources
        if resources_per_worker is None:
            resources_per_worker = {"num_cpus": 0.5}
        
        # Create Ray remote actors for each gamefile
        GamefileWorkerActor = ray.remote(**resources_per_worker)(GamefileBasedWorker)
        self.workers = []
        for gamefile in gamefiles:
            worker = GamefileWorkerActor.remote(gamefile)
            self.workers.append(worker)
        
        self.prev_admissible_commands = [None for _ in range(self.num_processes)]
    
    def step(self, actions):
        """Execute actions in all environments in parallel."""
        assert len(actions) == self.num_processes, \
            f"Number of actions ({len(actions)}) must equal number of environments ({self.num_processes})"
        
        # Send step commands to all workers
        futures = [worker.step.remote(action) for worker, action in zip(self.workers, actions)]
        
        # Collect results
        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []
        
        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            text_obs_list.append(obs[0] if isinstance(obs, (list, tuple)) else obs)
            dones_list.append(dones[0] if isinstance(dones, (list, tuple)) else dones)
            info_list.append(info)
            
            self.prev_admissible_commands[i] = info.get('admissible_commands', [])
            rewards_list.append(compute_reward(info, self.multi_modal))
        
        # No image obs for text-only environments
        image_obs_list = None
        
        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list
    
    def reset(self):
        """Reset all environments in parallel."""
        text_obs_list = []
        info_list = []
        
        # Send reset commands to all workers
        futures = [worker.reset.remote() for worker in self.workers]
        
        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            text_obs_list.append(obs[0] if isinstance(obs, (list, tuple)) else obs)
            self.prev_admissible_commands[i] = info.get('admissible_commands', [])
            info_list.append(info)
        
        # No image obs for text-only environments
        image_obs_list = None
        
        return text_obs_list, image_obs_list, info_list
    
    @property
    def get_admissible_commands(self):
        """Return the admissible commands stored from last step/reset."""
        return self.prev_admissible_commands
    
    def close(self):
        """Close all workers."""
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception:
                pass  # Worker may already be dead


def build_evolution_envs_alfworld(gamefiles, config, resources_per_worker=None):
    """
    Create an independent environment manager for evolution trajectory collection.
    
    This creates a completely separate set of environments that won't interfere
    with the main training loop's environments.
    
    Args:
        gamefiles: List of game file paths to load
        config: Configuration object (used by AlfWorldEnvironmentManager)
        resources_per_worker: Ray resource allocation per worker (optional)
        
    Returns:
        AlfWorldEnvironmentManager instance wrapping GamefileBasedAlfworldEnvs
        
    Example:
        >>> gamefiles = ["/path/to/game1.tw-pddl", "/path/to/game2.tw-pddl"]
        >>> envs = build_evolution_envs_alfworld(gamefiles, config)
        >>> obs, infos = envs.reset(kwargs=None)
        >>> next_obs, rewards, dones, infos = envs.step(actions)
        >>> envs.close()
    """
    from functools import partial
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import alfworld_projection
    
    _envs = GamefileBasedAlfworldEnvs(gamefiles, resources_per_worker)
    projection_f = partial(alfworld_projection)
    return AlfWorldEnvironmentManager(_envs, projection_f, config)
