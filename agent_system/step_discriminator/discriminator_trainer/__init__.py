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
Discriminator Trainer Module.

This module provides tools for training a step-level credit discriminator model
that determines whether each step in a trajectory is necessary or redundant.

Main components:
- data_processor: Convert credit analysis JSON to training Parquet files
- trainer: Train the discriminator model using verl FSDP SFT Trainer

Usage:
    # Step 1: Generate training data
    python -m agent_system.step_discriminator.discriminator_trainer.data_processor \\
        --input_file results/credit_analysis_data/credit_analysis.json \\
        --output_dir results/discriminator_data
    
    # Step 2: Train discriminator with verl FSDP
    torchrun --nproc_per_node=4 \\
        -m agent_system.step_discriminator.discriminator_trainer.trainer \\
        data.train_files=results/discriminator_data/train.parquet \\
        data.val_files=results/discriminator_data/val.parquet
"""

from .data_processor import (
    load_credit_analysis_file,
    build_prompt,
    build_trajectory_text,
    generate_samples,
    process_data,
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_WITH_OBS,
    PROMPT_TEMPLATE_ACTION_ONLY,
    VALID_ENV_TYPES,
)

__all__ = [
    "load_credit_analysis_file",
    "build_prompt",
    "build_trajectory_text",
    "generate_samples",
    "process_data",
    "PROMPT_TEMPLATE",
    "PROMPT_TEMPLATE_WITH_OBS",
    "PROMPT_TEMPLATE_ACTION_ONLY",
    "VALID_ENV_TYPES",
]
