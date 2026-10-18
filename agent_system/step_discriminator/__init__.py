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
Step Discriminator Module for Step-level Credit Analysis and Discriminator Training.

This module provides tools for:
1. Trajectory collection and credit analysis (trajectory_analyzer)
2. Discriminator model training (discriminator_trainer)

Submodules:
- trajectory_analyzer: Collect trajectories and analyze step credit levels
- discriminator_trainer: Train discriminator model to predict step necessity
"""

# Re-export from submodules for convenience
from .trajectory_analyzer import (
    CreditCheckResult,
    CreditAnalysisResult,
    CreditLevelChecker,
    analyze_trajectory_credit_level,
    CreditAnalyzer,
    run_analysis,
)

__all__ = [
    # Credit level analysis
    'CreditCheckResult',
    'CreditAnalysisResult', 
    'CreditLevelChecker',
    'analyze_trajectory_credit_level',
    # Batch analysis utilities
    'CreditAnalyzer',
    'run_analysis',
]
