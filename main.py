import random
import os
import warnings

import template
from data.instruction_induction.load_data import load_data, tasks
from exec_accuracy import exec_accuracy_evaluator
from config import PROMPT_SET, APE_PROMPT_SET, APE_PROMPTs, Negative_SET

import fire
import pandas as pd

warnings.filterwarnings("ignore")

def getPrompt(ori_prompt, num_str):
    new_prompt = ori_prompt
    if num_str > 0:
        new_prompt = ori_prompt + " " + Negative_SET[num_str - 1]
    return new_prompt

def run(task, model, pnum, few_shot):
    results = []
    results_dir = os.path.join(os.getcwd(), f'results/{model}')
    
    if task == 'all':
        task_list = tasks
    else:
        assert task in tasks, 'Task not found!'
        task_list = [task]
    
    if pnum == 'all':
        pnum_list = range(0, 11)  # 0 to 10, where 0 is no modification
    else:
        pnum_list = [pnum]
    
    for current_task in task_list:
        for current_pnum in pnum_list:
            for iteration in range(1, 6):  # 5 iterations
                test_data = load_data('eval', current_task)
                origin_prompt = PROMPT_SET[current_task]
                
                # few-shot setting
                induce_data = load_data('induce', current_task)
                few_shot_data = induce_data[0], [random.sample(output, 1)[0] for output in induce_data[1]]
                num_demos = 5
                demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
                eval_template = "Instruction: [PROMPT]\n\n[full_DEMO]\nInput: [INPUT]\nAnswer: [OUTPUT]"
                demos_template = template.DemosTemplate(demos_template)
                
                print(f'LLM: {model}, Task: {current_task}, Pnum: {current_pnum}, Iteration: {iteration}')
                
                new_prompt = getPrompt(origin_prompt, current_pnum)
                # print('Prompt: ', new_prompt)
                # print('Few_shot: ', few_shot)
                
                test_num = min(100, len(test_data[0]))
                
                eval_template = template.EvalTemplate(eval_template)
                test_res = exec_accuracy_evaluator(prompts=[new_prompt],
                                                   eval_template=eval_template,
                                                   eval_data=test_data,
                                                   llm_model=model,
                                                   pnum=current_pnum,
                                                   task=current_task,
                                                   num_samples=test_num,
                                                   few_shot=few_shot,
                                                   demos_template=demos_template,
                                                   few_shot_data=few_shot_data,
                                                   num_demos=num_demos)
                
                test_score = test_res.sorted()[1][0]
                print(f'Test score for "{current_task} with pnum {current_pnum}": {test_score}\n\n')
                
                # Add result to the list
                results.append({
                    'task': current_task,
                    'pnum': current_pnum,
                    'iteration': iteration,
                    'score': test_score
                })
                
                # Create a DataFrame and save partial results
                df = pd.DataFrame(results)
                os.makedirs(results_dir, exist_ok=True)
                path = os.path.join(results_dir, f'{current_task}_{current_pnum}_partial.csv')
                df.to_csv(path, index=False)
    
    # Save final results
    final_df = pd.DataFrame(results)
    path = os.path.join(results_dir, f'{task}_{pnum}.csv')
    final_df.to_csv(path, index=False)
    # Remove all partial result files
    # for file in os.listdir(results_dir):
    #     if file.endswith('_partial.csv'):
    #         os.remove(os.path.join(results_dir, file))

if __name__ == '__main__':
    fire.Fire(run)