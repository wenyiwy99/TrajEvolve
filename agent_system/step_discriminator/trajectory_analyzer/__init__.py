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
Trajectory Analyzer Module for SFT Data Construction.

This module provides tools for:
1. Analyzing step credit levels (necessary vs redundant)
2. Generating training data for discriminator model

Main components:
- credit_analyzer: Analyze trajectory step credit levels

Usage:
    python -m agent_system.step_discriminator.trajectory_analyzer.credit_analyzer \\
        env_type=alfworld \\
        input_file=/path/to/trajectories.json \\
        output_dir=/path/to/output
"""

from .credit_analyzer import (
    CreditCheckResult,
    CreditAnalysisResult,
    CreditLevelChecker,
    analyze_trajectory_credit_level,
    # Batch analysis utilities
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
