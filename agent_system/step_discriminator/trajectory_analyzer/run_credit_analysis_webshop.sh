#!/bin/bash
# Run credit analysis on pre-collected WebShop trajectories

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../credit_analysis_data/webshop"

python3 -u -m agent_system.step_discriminator.trajectory_analyzer.credit_analyzer \
    env_type=webshop \
    input_file=${DATA_DIR}/trajectories.json \
    output_dir=${DATA_DIR}
