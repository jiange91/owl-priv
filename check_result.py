import json
import numpy as np
from collections import defaultdict

def inspect(fpath, first_n=None):
    with open(fpath, 'r') as f:
        data = json.load(f)
    scores = []
    tasks = []
    for task in data:
        scores.append(task['score'])
        if not task['score'] and (first_n is None or len(tasks) < first_n):
            tasks.append(task)
    avg_score = np.mean(scores)
    print(f'num_tasks: {len(data)}')
    print(f'Average score: {avg_score}')
    
    if tasks:
        with open('failed_tasks.json', 'w') as f:
            json.dump(tasks, f, indent=4)
    
def ref_inspect(fpath, dump_fail=False, dump_levels=None):
    with open(fpath, 'r') as f:
        es = f.readlines()
    scores = defaultdict(list)
    data = []
    for e in es:
        data_json = json.loads(e)
        data.append(data_json)
        if 'level' not in data_json:
            level = 0
        else:
            level = data_json['level']
        scores[level].append(data_json['score'])
    for level, score_list in scores.items():
        avg_score = np.mean(score_list)
        print(f'Level: {level}, Average score: {avg_score*100:.2f}%, Count: {sum(score_list)}/{len(score_list)}')
    avg_score = np.mean([s for sublist in scores.values() for s in sublist])
    print(f'num_tasks: {len(es)}')
    print(f'Average score: {avg_score}')
    if dump_fail:
        if dump_levels is None:
            dump_levels = [1, 2, 3]
        fail_tasks = [task for task in data if not task['score'] and task['level'] in dump_levels]
        with open('ref_fail_tasks.json', 'w') as f:
            json.dump(fail_tasks, f, indent=4)
    
def calc_dff(ref_path, run_path, prefix, dump_missmatch_id, dump_ref_subset, first_n):
    with open(ref_path, 'r') as f:
        data = f.readlines()
    ref_data = [json.loads(e) for e in data]
    task_id_2_ref = {task['task_id']: task for task in ref_data}
    
    with open(run_path, 'r') as f:
        run_data = json.load(f)

    missmatches = []
    task_id_2_run = {task['task_id']: task for task in run_data}
    for task in run_data:
        task_id = task['task_id']
        if task_id not in task_id_2_ref:
            print(f'Task {task_id} not found in reference data.')
            continue
        ref_task = task_id_2_ref[task_id]
        if task['score'] != ref_task['score']:
            print(f'Task {task_id} score mismatch run vs ref: {task["score"]}, {ref_task["score"]}')
            missmatches.append((task_id, task['score'], ref_task['score']))
    if first_n is not None:
        missmatches = missmatches[:first_n]
    print(f'Number of mismatches: {len(missmatches)}')
    
    # create directory if it doesn't exist
    import os
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    if dump_missmatch_id:
        with open(f'{prefix}/mismatches.json', 'w') as f:
            json.dump(missmatches, f, indent=4)
        print(f'Mismatches dumped to mismatches.json')
        
    if dump_ref_subset:
        ref_missmatchs = [task_id_2_ref[task_id] for task_id, _, _ in missmatches]
        with open(f'{prefix}/ref_subset.json', 'w') as f:
            json.dump(ref_missmatchs, f, indent=4)
        print(f'Reference subset dumped to ref_subset.json')
    
    task_subset = [task_id_2_run[task_id] for task_id, _, _ in missmatches]
    with open(f'{prefix}/task_subset.json', 'w') as f:
        json.dump(task_subset, f, indent=4)

# inspect('/mnt/ssd3/zijian-proj/owl-priv/results/workforce/workforce_2_pass1_mini_l2.json', first_n=0)
ref_inspect('/mnt/ssd3/zijian-proj/owl-priv/workforce_claude.jsonl', dump_fail=False)
# calc_dff(
#     '/mnt/ssd3/zijian-proj/owl-priv/workforce_claude.jsonl',
#     '/mnt/ssd3/zijian-proj/owl-priv/results/workforce/workforce_2_pass1_mini_l2.json',
#     'workforce_2_pass1_mini_l2',
#     dump_missmatch_id=True,
#     dump_ref_subset=True,
#     first_n=10
# )