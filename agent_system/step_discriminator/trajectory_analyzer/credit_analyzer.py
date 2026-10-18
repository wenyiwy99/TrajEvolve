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
"""
Credit Level Checker: Analyze step credit levels via ablation.

This module determines whether each step in a trajectory is necessary or redundant
for task completion by verifying in the environment. Dependent steps are determined
by action validity (admissible_commands for ALFWorld, available_actions for WebShop).
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from agent_system.environments.env_package.webshop.envs import WebshopWorker


def create_single_webshop_env(seed: int = 0, env_config=None):
    """
    Create a single WebShop environment instance using WebshopWorker wrapper.

    Args:
        seed: Random seed for reproducible product prices etc.
        env_config: Environment configuration (optional)

    Returns:
        WebshopWorker instance with unified step/reset/close interface
    """
    env_kwargs = {
        'observation_mode': 'text',
        'human_goals': False,
        'num_prev_obs': 0,
        'num_prev_actions': 0,
    }
    env = WebshopWorker(seed=seed, env_kwargs=env_kwargs)
    return env


@dataclass
class CreditCheckResult:
    """Result of analyzing a single step's credit level."""
    step_number: int
    action_text: str
    observation_text: str
    is_action_valid: bool
    credit_level: str  # 'necessary', 'redundant', or 'skipped_invalid'
    dependent_steps: List[int] = field(default_factory=list)
    env_verification_result: Optional[bool] = None


@dataclass
class CreditAnalysisResult:
    """Complete analysis result for a trajectory."""
    trajectory_info: Dict
    step_analyses: List[CreditCheckResult]
    necessary_count: int = 0
    redundant_count: int = 0
    invalid_count: int = 0


class CreditLevelChecker:
    """
    Analyze step credit levels using ablation and action-validity-based dependency detection.

    Algorithm:
    1. For each step i, skip it (along with previously confirmed redundant steps)
    2. Execute remaining trajectory in environment
    3. For each step j > i, check if its action is still valid given current state
       - ALFWorld: check if observation contains "Nothing happens"
       - WebShop: check if action is valid using available_actions
    4. If action becomes invalid, the step is added to dependent_steps
    5. If step i is redundant, all steps in dependent_steps are also marked redundant
    6. previously_redundant accumulates: when analyzing later steps,
       earlier redundant steps are always skipped
    """

    def __init__(self, env_manager=None, config=None, env_type: str = 'alfworld'):
        """
        Initialize the credit level checker.

        Args:
            env_manager: Environment manager for verification (optional)
            config: Configuration object (optional)
            env_type: Environment type, 'alfworld' or 'webshop' (default: 'alfworld')
        """
        self.env_manager = env_manager
        self.config = config
        self.env_type = env_type

    def verify_in_environment(self, trajectory: Dict, skipped_indices: List[int], step_index: int) -> Tuple[bool, List[int]]:
        """
        Verify if the trajectory still succeeds when certain steps are skipped,
        and compute dependent_steps based on action validity.

        Returns:
            Tuple[bool, List[int]]: (env_success, dependent_steps)
            - env_success: Whether the task was completed successfully
            - dependent_steps: Steps after step_index whose actions become invalid
        """
        if self.env_type == 'webshop':
            return self._verify_in_webshop_environment(trajectory, skipped_indices, step_index)
        else:
            return self._verify_in_alfworld_environment(trajectory, skipped_indices, step_index)

    def _verify_in_alfworld_environment(self, trajectory: Dict, skipped_indices: List[int], step_index: int) -> Tuple[bool, List[int]]:
        """
        Verify trajectory in ALFWorld environment and compute dependent_steps.

        Executes non-skipped steps in order. For steps after step_index, checks if
        observation contains "Nothing happens" to detect invalid actions.

        Returns:
            Tuple[bool, List[int]]: (won, dependent_steps)
        """
        from agent_system.environments.env_package.alfworld import create_single_game_env

        gamefile = trajectory.get('gamefile')
        steps = trajectory.get('trajectory', [])
        skipped_set = set(skipped_indices)
        steps_to_check = set(j for j in range(step_index + 1, len(steps)) if j not in skipped_set)

        single_env = create_single_game_env(gamefile)

        try:
            obs, info = single_env.reset()

            won = False
            dependent_steps = []

            for i, step in enumerate(steps):
                if i in skipped_set:
                    continue

                action = step.get('action_text', '')
                obs, score, done, info = single_env.step(action)

                if i in steps_to_check:
                    if "Nothing happens" in obs:
                        dependent_steps.append(i)

                if info.get('won', False):
                    won = True
                    break

                if done:
                    break

            return won, dependent_steps

        finally:
            single_env.close()

    @staticmethod
    def _is_action_valid_webshop(action: str, available_actions: dict) -> bool:
        """
        Check if an action is valid in the current WebShop state.

        Mirrors the validity logic in WebAgentTextEnv.step():
        - search[keywords]: valid when has_search_bar=True, keywords is non-empty
        - click[value]: valid when value is in clickables and value != 'search'
        - Otherwise: invalid

        Args:
            action: Action string, e.g. "search[red shoes]" or "click[buy now]"
            available_actions: Dict with 'has_search_bar' (bool) and 'clickables' (list)

        Returns:
            True if the action would be executed by the environment, False otherwise
        """
        pattern = re.compile(r'(.+)\[(.+)\]')
        m = re.match(pattern, action)
        if m is None:
            return False

        action_name, action_arg = m.groups()
        action_arg = action_arg.lower()

        if (action_name == 'search' and
            action_arg is not None and
            action_arg != ''):
            return available_actions.get('has_search_bar', False)
        elif (action_name == 'click' and
              action_arg != 'search'):
            clickables = available_actions.get('clickables', [])
            return action_arg in clickables

        return False

    def _verify_in_webshop_environment(self, trajectory: Dict, skipped_indices: List[int], step_index: int) -> Tuple[bool, List[int]]:
        """
        Verify trajectory in WebShop environment and compute dependent_steps.

        Reconstructs the environment precisely using env_seed and session_id.
        WebshopWorker wrapper automatically adds info['won'] and other fields.
        For steps after step_index, checks action validity using available_actions.

        Returns:
            Tuple[bool, List[int]]: (won, dependent_steps)
        """
        env_seed = trajectory.get('env_seed')
        session_id = trajectory.get('session_id')

        if env_seed is None or session_id is None:
            raise ValueError(
                f"Trajectory missing env_seed or session_id for seed-based reconstruction. "
                f"env_seed={env_seed}, session_id={session_id}. "
                f"Please re-collect trajectories using collector_webshop.py."
            )

        steps = trajectory.get('trajectory', [])
        skipped_set = set(skipped_indices)
        steps_to_check = set(j for j in range(step_index + 1, len(steps)) if j not in skipped_set)

        single_env = create_single_webshop_env(seed=env_seed)
        try:
            obs, info = single_env.reset(int(session_id))

            won = False
            dependent_steps = []

            for i, step in enumerate(steps):
                if i in skipped_set:
                    continue

                action = step.get('action_text', '')

                if i in steps_to_check:
                    pre_step_available = single_env.get_available_actions()
                    if not self._is_action_valid_webshop(action, pre_step_available):
                        dependent_steps.append(i)

                obs, reward, done, info = single_env.step(action)

                if info.get('won', False):
                    won = True
                    break

                if done:
                    break

            return won, dependent_steps

        finally:
            single_env.close()

    def analyze_trajectory(self, trajectory: Dict) -> CreditAnalysisResult:
        """
        Analyze all steps in a trajectory for credit level.

        When a step is redundant, all steps in its dependent_steps are also marked redundant.
        """
        steps = trajectory.get('trajectory', [])
        trajectory_info = {
            'task': trajectory.get('task'),
            'total_steps': len(steps),
            'is_success': trajectory.get('is_success', False),
            'traj_uid': trajectory.get('traj_uid'),
        }

        traj_id = trajectory.get('task_id', trajectory.get('gamefile', 'unknown'))

        step_analyses: List[CreditCheckResult] = []
        already_analyzed: Dict[int, CreditCheckResult] = {}
        previously_redundant: List[int] = []

        print(f"\nAnalyzing trajectory with {len(steps)} steps...")
        print(f"Trajectory ID: {traj_id}")
        print("=" * 60)

        for i in range(len(steps)):
            if i in already_analyzed:
                continue

            current_step = steps[i]
            action_text = current_step.get('action_text', '')
            observation_text = current_step.get('observation_text', '')
            is_valid = current_step.get('is_action_valid', False)

            if not is_valid:
                result = CreditCheckResult(
                    step_number=i,
                    action_text=action_text,
                    observation_text=observation_text,
                    is_action_valid=False,
                    credit_level='skipped_invalid',
                    dependent_steps=[],
                    env_verification_result=None,
                )
            else:
                all_skipped = [i] + list(previously_redundant)
                env_success, dependent_steps = self.verify_in_environment(trajectory, all_skipped, i)
                credit_level = 'redundant' if env_success else 'necessary'

                result = CreditCheckResult(
                    step_number=i,
                    action_text=action_text,
                    observation_text=observation_text,
                    is_action_valid=True,
                    credit_level=credit_level,
                    dependent_steps=dependent_steps,
                    env_verification_result=env_success,
                )

            step_analyses.append(result)
            already_analyzed[i] = result

            if result.credit_level in ('redundant', 'skipped_invalid'):
                previously_redundant.append(i)

            if result.credit_level == 'redundant':
                for chain_idx in result.dependent_steps:
                    step = steps[chain_idx]
                    is_valid = step.get('is_action_valid', False)
                    credit = 'skipped_invalid' if not is_valid else 'redundant'

                    chain_result = CreditCheckResult(
                        step_number=chain_idx,
                        action_text=step.get('action_text', ''),
                        observation_text=step.get('observation_text', ''),
                        is_action_valid=is_valid,
                        credit_level=credit,
                        dependent_steps=[],
                        env_verification_result=result.env_verification_result,
                    )

                    if chain_idx in already_analyzed:
                        already_analyzed[chain_idx].credit_level = credit
                    else:
                        step_analyses.append(chain_result)
                        already_analyzed[chain_idx] = chain_result

                    if credit == 'redundant' and chain_idx not in previously_redundant:
                        previously_redundant.append(chain_idx)

        step_analyses.sort(key=lambda r: r.step_number)

        necessary_count = sum(1 for r in step_analyses if r.credit_level == 'necessary')
        redundant_count = sum(1 for r in step_analyses if r.credit_level == 'redundant')
        invalid_count = sum(1 for r in step_analyses if r.credit_level == 'skipped_invalid')

        print("\n" + "=" * 60)
        print(f"Analysis complete:")
        print(f"  Necessary steps: {necessary_count}")
        print(f"  Redundant steps: {redundant_count}")
        print(f"  Invalid steps: {invalid_count}")

        return CreditAnalysisResult(
            trajectory_info=trajectory_info,
            step_analyses=step_analyses,
            necessary_count=necessary_count,
            redundant_count=redundant_count,
            invalid_count=invalid_count,
        )

    def to_dict(self, result: CreditAnalysisResult) -> Dict:
        """Convert analysis result to a serializable dict."""
        return {
            'analysis_method': 'hindsight verification',
            'trajectory_info': result.trajectory_info,
            'step_analysis': [
                {
                    'step': r.step_number,
                    'action': r.action_text,
                    'observation': r.observation_text,
                    'is_action_valid': r.is_action_valid,
                    'credit_level': r.credit_level,
                    'dependent_steps': r.dependent_steps,
                }
                for r in result.step_analyses
            ],
        }


def analyze_trajectory_credit_level(
    trajectory: Dict,
    env_manager=None,
    config=None,
    env_type: str = 'alfworld',
) -> Dict:
    """
    Analyze trajectory step credit levels.

    This is a convenience entry function for direct calling.

    Args:
        trajectory: Trajectory data dict containing 'trajectory' list
        env_manager: Environment manager (optional)
        config: Configuration object (optional)
        env_type: Environment type, 'alfworld' or 'webshop' (default: 'alfworld')

    Returns:
        Analysis result dict with 'analysis_method', 'trajectory_info', 'step_analysis'
    """
    checker = CreditLevelChecker(env_manager, config, env_type=env_type)
    result = checker.analyze_trajectory(trajectory)
    return checker.to_dict(result)


# ============================================================
# Batch Analysis Utilities
# ============================================================

class CreditAnalyzer:
    """
    Analyzer class for credit level analysis on pre-collected trajectories.

    Loads trajectories from JSON files and performs parallel analysis using Ray.
    """

    def __init__(self, config):
        """
        Initialize the credit analyzer.

        Args:
            config: Configuration dict or OmegaConf object with:
                - env_type: Environment type ('alfworld' or 'webshop')
                - input_file: Path to trajectories.json
                - output_dir: Directory to save credit_analysis.json
        """
        self.config = config
        self.env_type = config.get('env_type', 'alfworld')
        self.input_file = config.get('input_file')
        self.output_dir = config.get('output_dir')

    def load_trajectories(self, json_path: str = None) -> List[Dict]:
        """
        Load trajectories from a JSON file.

        Expected JSON structure:
        {
            "config": {...},
            "results": [
                {"is_success": true, "trajectory": [...], ...},
                ...
            ]
        }

        Args:
            json_path: Path to the trajectories JSON file (optional, uses config if not provided)

        Returns:
            List of trajectory dicts
        """
        if json_path is None:
            json_path = self.input_file

        if not json_path:
            raise ValueError("No input file specified")

        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Trajectories file not found: {json_path}")

        print(f"Loading trajectories from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'results' in data:
            trajectories = data['results']
        elif isinstance(data, list):
            trajectories = data
        else:
            raise ValueError(f"Unexpected data format in {json_path}")

        print(f"Loaded {len(trajectories)} trajectories")
        return trajectories

    def analyze_trajectories_parallel(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Analyze trajectories in parallel using Ray.

        Args:
            trajectories: List of trajectory dicts to analyze

        Returns:
            List of analysis results
        """
        import ray

        if not trajectories:
            print("No trajectories to analyze")
            return []

        @ray.remote
        def _analyze_remote(traj: Dict, env_type: str) -> Dict:
            return analyze_trajectory_credit_level(traj, env_type=env_type)

        print(f"\nAnalyzing {len(trajectories)} trajectories in parallel (env_type={self.env_type})...")

        futures = [_analyze_remote.remote(t, self.env_type) for t in trajectories]
        results = ray.get(futures)
        valid_results = [r for r in results if r is not None]

        print(f"Completed analysis of {len(valid_results)} trajectories")
        return valid_results

    def save_analysis_results(self, analysis_results: List[Dict], output_dir: str = None) -> str:
        """
        Save credit analysis results to JSON file.

        Args:
            analysis_results: List of analysis results
            output_dir: Directory to save results (optional, uses config if not provided)

        Returns:
            Path to the saved file
        """
        if output_dir is None:
            output_dir = self.output_dir

        if not output_dir:
            raise ValueError("No output directory specified")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = output_path / "credit_analysis.json"

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'env_type': self.env_type,
                    'input_file': str(self.input_file),
                    'timestamp': timestamp,
                },
                'analysis_results': analysis_results
            }, f, indent=2, ensure_ascii=False)

        print(f"\nAnalysis results saved to: {analysis_file}")
        print(f"  Trajectories analyzed: {len(analysis_results)}")

        return str(analysis_file)

    def run(self) -> Dict:
        """
        Run the complete analysis pipeline.

        Returns:
            Dict with output_path and analysis summary
        """
        trajectories = self.load_trajectories()
        analysis_results = self.analyze_trajectories_parallel(trajectories)
        output_path = self.save_analysis_results(analysis_results)

        return {
            "output_path": output_path,
            "num_trajectories": len(analysis_results),
        }


def run_analysis(config) -> Dict:
    """
    Main entry point for running credit analysis from command line.

    Args:
        config: Configuration dict or OmegaConf object with:
            - env_type: Environment type ('alfworld' or 'webshop')
            - input_file: Path to trajectories.json
            - output_dir: Directory to save credit_analysis.json

    Returns:
        Dict with output_path, num_trajectories, and analysis summary
    """
    import ray

    if ray.is_initialized():
        print("Ray already initialized, shutting down to restart...")
        ray.shutdown()

    print("Initializing Ray...")
    ray.init(
        runtime_env={
            "excludes": ["/.git/", "*.txt", "*.log", "ray_submit_log/"],
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
            }
        },
        num_cpus=config.get('num_cpus', None),
    )

    analyzer = CreditAnalyzer(config)
    result = analyzer.run()

    return result


if __name__ == '__main__':
    from omegaconf import OmegaConf

    cli_config = OmegaConf.from_cli()

    default_config = OmegaConf.create({
        'env_type': 'alfworld',
        'input_file': None,
        'output_dir': None,
        'num_cpus': None,
    })

    config = OmegaConf.merge(default_config, cli_config)

    if not config.input_file:
        raise ValueError("input_file is required. Usage: python -m ... input_file=/path/to/trajectories.json")
    if not config.output_dir:
        raise ValueError("output_dir is required. Usage: python -m ... output_dir=/path/to/output")

    result = run_analysis(config)
    print(f"\nDone! Output: {result['output_path']}")
