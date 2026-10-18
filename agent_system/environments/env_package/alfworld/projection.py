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

from typing import List
import re

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0
            
        if valids[i] == 0:
            actions[i] = actions[i] + "]"
    return actions, valids


def map_to_admissible_alfworld(action: str, admissible_commands: List[str]) -> str:
    """
    将单个 action 映射到最匹配的 admissible command。
    
    匹配逻辑：
    1. 完全匹配
    2. 检查是否有 admissible command 包含在 action 中
    3. 最长公共子串匹配
    
    Args:
        action: 原始动作文本
        admissible_commands: 可选动作列表
        
    Returns:
        最匹配的 admissible command
    """
    if not admissible_commands:
        return action
    
    action_lower = action.lower().strip()
    
    # Step 1: 完全匹配
    for cmd in admissible_commands:
        if cmd.lower() == action_lower:
            return cmd
    
    # Step 2: 检查是否有 admissible command 包含在 action 中
    for cmd in admissible_commands:
        if cmd.lower() in action_lower:
            return cmd
    
    # Step 3: 最长公共子串匹配
    def lcs_length(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(2)]
        max_len = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
                    max_len = max(max_len, dp[i % 2][j])
                else:
                    dp[i % 2][j] = 0
        return max_len
    
    best_cmd = admissible_commands[0]
    best_score = 0
    for cmd in admissible_commands:
        score = lcs_length(action_lower, cmd.lower())
        if score > best_score:
            best_score = score
            best_cmd = cmd
    
    return best_cmd
