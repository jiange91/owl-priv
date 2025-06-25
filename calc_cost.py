from openai_function_tokens import estimate_tokens
from typing import Any

from run_gaia_workforce_cheap import construct_workforce
# from run_gaia_workforce_claude import construct_workforce
from utils import OwlWorkforceChatAgent, OwlGaiaWorkforce, OwlSingleAgentWorker
from camel.types import(
    ModelPlatformType,
    ModelType
)

workforce = construct_workforce()
# print(agents[0]["agent"]._get_full_tool_schemas())
agent_2_functions: dict[str, list[dict]] = {}
agent_2_agent: dict[str, OwlWorkforceChatAgent] = {}
for node in workforce._children:
    node: OwlSingleAgentWorker
    agent = node.worker
    name = node.name
    agent_2_functions[name] = [e['function'] for e in agent._get_full_tool_schemas()]
    agent_2_agent[name] = agent

    
msgs = [
    {
        "role": "system",
        "content": "You are a helpful assistant",
    },
    {
        "role": "user",
        "content": "We are solving a complex task",
    },
    {
        "role": "assistant",
        "content": "",
        "function_call": {
            "name": "search_wiki",
            "arguments": "{\"entity\": \"vegetable\"}"
        }
    },
]

# print(estimate_tokens(
#     messages=msgs,
#     functions=agent_2_functions['Web Agent']
# ))

# per 1M token
prefill_cost_table = {
    ModelType.GPT_4_1: 2.0,
    ModelType.GPT_4_1_MINI: 0.4,
    ModelType.GPT_4O: 2.5,
    ModelType.GPT_4O_MINI: 0.15,
    ModelType.O3_MINI: 1.1,
    ModelType.CLAUDE_3_7_SONNET: 3,
}
decode_cost_table = {
    ModelType.GPT_4_1: 8,
    ModelType.GPT_4_1_MINI: 1.6,
    ModelType.GPT_4O: 10,
    ModelType.GPT_4O_MINI: 0.6,
    ModelType.O3_MINI: 4.4,
    ModelType.CLAUDE_3_7_SONNET: 15,
}
M = 1_000_000

def calc_convo_cost(conversation: list[dict], model: ModelType, agent_name: str):
    rolling_context = []
    cost = 0
    for message in conversation:
        role = message['role']
        message['content'] = message['content'][:200_000]
        if role == 'assistant':
            # calculate prompt token
            # print(agent_2_functions.get(agent_name))
            prompt_token = estimate_tokens(
                messages=rolling_context,
                functions=agent_2_functions.get(agent_name),
            )
            # reformat function call
            if "tool_calls" in message:
                message['function_call'] = message['tool_calls'][0]["function"]
            # calculate response token
            output_token = estimate_tokens(
                messages=[message],
            )
            cost += prompt_token / M * prefill_cost_table[model] + output_token / M * decode_cost_table[model]
            # add to prompt
            rolling_context.append(message)
    return cost

def calc_plan_cost(plan: dict[str, Any]):
    worker_history = plan.get('subtasks_history', [])
    planner_history = plan.get('planner_history', [])
    coordinator_history = plan.get('coordinator_history', [])
    
    cost = 0
    for subtask in worker_history:
        model = agent_2_agent[subtask['assignee']].model_type
        convo = subtask.get('trajectory', [])
        cost += calc_convo_cost(conversation=convo, model=model, agent_name=subtask['assignee'])
    
    planner_model = workforce.task_agent.model_type
    cost += calc_convo_cost(conversation=planner_history, model=planner_model, agent_name='Task Agent')
    
    coordinator_model = workforce.coordinator_agent.model_type
    cost += calc_convo_cost(conversation=coordinator_history, model=coordinator_model, agent_name='Coordinator Agent')
    return cost

def calc_task_cost(task: dict[str, Any]):
    attempts = task.get('trajectory', [])
    total_cost = 0
    for attempt in attempts:
        plans = attempt.get('trajectory', [])
        for plan in plans:
            total_cost += calc_plan_cost(plan)
            
    return total_cost

import json
with open('/mnt/ssd3/zijian-proj/owl-priv/results/workforce/workforce_1_pass1_debug_search.json', 'r') as f:
    data = json.load(f)

success = [e for e in data if e['score']]
prices = [calc_task_cost(e) for e in success]

import numpy as np
print(np.mean(prices))