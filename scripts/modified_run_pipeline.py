import argparse
import yaml
import os
import csv
import subprocess
import re
import torch
import signal
import ray
import random
import numpy as np 
from tqdm import tqdm
import sys
sys.path.extend(['./baselines', '../baselines'])
from model_utils import _init_ray

# (non-exhaustive) methods list: AutoDAN,AutoPrompt,DirectRequest,GCG-Multi,GCG-Transfer,FewShot,GBDA,GCG,HumanJailbreaks,PAIR,PAP-top5,PEZ,TAP,TAP-Transfer,UAT,ZeroShot
# (non-exhaustive) models list: llama2_7b,llama2_13b,llama2_70b,vicuna_7b_v1_5,vicuna_13b_v1_5,koala_7b,koala_13b,orca_2_7b,orca_2_13b,solar_10_7b_instruct,openchat_3_5_1210,starling_7b,mistral_7b_v2,mixtral_8x7b,zephyr_7b_robust6_data2,zephyr_7b,baichuan2_7b,baichuan2_13b,qwen_7b_chat,qwen_14b_chat,qwen_72b_chat,gpt-3.5-turbo-1106,gpt-3.5-turbo-0613,gpt-4-1106-preview,gpt-4-0613,gpt-4-vision-preview,claude-instant-1,claude-2.1,claude-2,gemini,mistral-medium,LLaVA,InstructBLIP,Qwen,GPT4V

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming pipeline with baseline methods for all test cases.")
    # Arguments for all steps
    parser.add_argument("--base_save_dir", type=str, default='./results',
                    help="The base directory for saving test cases, completions, and results for a specific set of behaviors")
    parser.add_argument("--base_log_dir", type=str, default='./slurm_logs',
                    help="The base directory for saving slurm logs")
    parser.add_argument("--pipeline_config_path", type=str, default='./configs/pipeline_configs/run_pipeline.yaml',
                    help="The path to the pipeline config file used to schedule/run jobs")
    parser.add_argument("--models_config_path", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--methods", type=str, default='GCG',
                        help="A list of red teaming method names (choices are in run_pipeline.yaml)")
    parser.add_argument("--models", type=str, default='llama2_7b',
                        help="A list of target models (from models.yaml)")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmful_behaviors_pair_custom.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--step", type=str, default="1",
                        help="The step of the pipeline to run",
                        choices=["1", "1.5", "2", "3", "2_and_3", "all"])
    parser.add_argument("--mode", type=str, default="slurm",
                        help="Whether to run the pipeline in 'local', 'slurm', or 'local_parallel' mode",
                        choices=["local", "slurm", "local_parallel"])
    parser.add_argument("--partition", type=str, default="amd_a100nv_8",
                        help="The partition to use for submitting slurm jobs")
    
    # Arguments for step 1 (test case generation)
    parser.add_argument("--behavior_ids_subset", type=str, default='',
                        help="An optional comma-separated list of behavior IDs, or a path to a newline-separated list of behavior IDs. If provided, test cases will only be generated for the specified behaviors. (only used in step 1 of the pipeline)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing saved test cases (only used in step 1 of the pipeline)")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print out intermediate results while generating test cases (only used in step 1 of the pipeline)")
    parser.add_argument("--extract_mode", type=str, default="all", 
                        choices=['all', 'last', 'best'],
                        help="Mode for extracting test cases: 'all' for all test cases, 'last' for final test case only, 'best' for lowest loss test case (step 1.5)")
    
    # Arguments for step 2 (completion generation)
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="The maximum number of new tokens to generate for each completion (only used in step 2 of the pipeline)")
    parser.add_argument("--incremental_update", action="store_true",
                        help="Whether to incrementally update completions or generate a new completions file from scratch  (only used in step 2 of the pipeline)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation (will be auto-adjusted if OOM) (only used in step 2 of the pipeline)")
    parser.add_argument("--generate_with_vllm", action="store_true",
                        help="Whether to generate completions with vLLM (if applicable) (only used in step 2 of the pipeline)")
    
    # Max length regeneration arguments for step 2
    parser.add_argument("--enable_max_length_regeneration", action="store_true", default=False,
                        help="Enable max length regeneration for AdvBench successful cases (only used in step 2 of the pipeline)")
    parser.add_argument("--max_regeneration_tokens", type=int, default=256,
                        help="Maximum tokens for regeneration when AdvBench succeeds (only used in step 2 of the pipeline)")
    
    # Arguments for step 3 (evaluation)
    parser.add_argument("--cls_path", type=str, default="/home/work/kisti/LLM/HarmBench-Llama-2-13b-cls",
                        help="The path to the classifier used for evaluating completions (only used in step 3 of the pipeline)")
    parser.add_argument("--use_advbench_first", action="store_true", default=True,
                        help="Whether to use AdvBench metric first, then classifier only on successful cases (only used in step 3 of the pipeline)")
    parser.add_argument("--include_advbench_metric", action="store_true", default=True,
                        help="Whether to include the AdvBench refusal metric (only used in step 3 of the pipeline)")
    parser.add_argument("--num_tokens", type=int, default=256,
                        help="The number of tokens to evaluate (only used in step 3 of the pipeline)")
    parser.add_argument("--save_first_success_only", action="store_true", default=False,
                        help="Save only the first successful test case for each behavior (only used in step 3 of the pipeline)")
    
    # General arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for full reproducibility")
    parser.add_argument("--gpus", type=str, default='0',
                    help="Comma-separated list of GPU IDs to use (e.g., '0,1'). If not set, use all available GPUs.")

    args = parser.parse_args()
    return args


def get_expanded_pipeline_configs(pipeline_config, model_configs, proper_method_names, models):
    """
    Expand the pipeline config to include the expected target models and experiment names for each method, as well as
    a mapping from target model to experiment name.
    
    Additional context: Each method config in HarmBench can include experiments that constitute multiple proper methods. E.g.,
    the test cases for GCG-Transfer are obtained from an instance of the EnsembleGCG class, which is also used to obtain
    test cases for GCG-Multi. The pipeline config file is used to specify proper methods and the experiments they correspond to.
    This function expands the pipeline config into a dictionary of method pipeline configs for each proper method, which contain
    the information necessary to submit jobs for the experiments corresponding to the specified target models. These method configs
    also help with analyzing the results of experiments.

    :param pipeline_config: The pipeline config dictionary
    :param model_configs: The model config dictionary
    :param proper_method_names: A list of proper method names (keys of pipeline_config)
    :param models: A list of model names (keys of the model config in models.yaml)
    :return: A dictionary of expanded pipeline configs for the specified proper methods and models.
    """
    expanded_pipeline_configs = {}
    for proper_method_name in proper_method_names:
        expanded_pipeline_config = pipeline_config[proper_method_name]
        experiment_name_template = expanded_pipeline_config['experiment_name_template']
        expanded_pipeline_config['experiments'] = []
        expanded_pipeline_config['model_to_experiment'] = {}
        if '<model_name>' in experiment_name_template:
            for model in models:
                if model_configs[model]['model_type'] not in expanded_pipeline_config['allowed_target_model_types']:
                    continue  # Skip models that are not allowed for this method
                experiment_name = experiment_name_template.replace('<model_name>', model)
                expanded_pipeline_config['experiments'].append({
                    'experiment_name': experiment_name,
                    'expected_target_models': [model],
                    'experiment_num_gpus': model_configs[model].get('num_gpus', 0),  # closed-source model configs don't have the num_gpus field, so we default to 0
                    'behavior_chunk_size': expanded_pipeline_config.get('behavior_chunk_size', 5)
                    })
                expanded_pipeline_config['model_to_experiment'][model] = experiment_name
        else:
            experiment_name = experiment_name_template
            expanded_pipeline_config['experiments'].append({
                'experiment_name': experiment_name,
                'expected_target_models': models,
                'experiment_num_gpus': 0,
                'behavior_chunk_size': expanded_pipeline_config.get('behavior_chunk_size', 5)
                })
            for model in models:
                if model_configs[model]['model_type'] not in expanded_pipeline_config['allowed_target_model_types']:
                    continue  # Skip models that are not allowed for this method
                expanded_pipeline_config['model_to_experiment'][model] = experiment_name

        expanded_pipeline_configs[proper_method_name] = expanded_pipeline_config
    
    return expanded_pipeline_configs


def run_subprocess_slurm(command):
    # Execute the sbatch command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the output and error, if any
    print("command:", command)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # Get the job ID for linking dependencies
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
    else:
        job_id = None

    return job_id

@ray.remote
def run_subprocess_ray(command):
    # Set the visible gpus for each command where resources={'visible_gpus': [(id, num_gpus)], ....}
    resources = ray.get_runtime_context().worker.core_worker.resource_ids()['visible_gpus']
    gpu_ids = [str(r[0]) for r in resources]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)

    print(f"gpus_ids={gpu_ids} | command={command}")
    process = subprocess.Popen(command, shell=True, env=env, preexec_fn=os.setsid)
    signal.signal(signal.SIGTERM, lambda signum, frame: os.killpg(os.getpgid(process.pid), signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda signum, frame: os.killpg(os.getpgid(process.pid), signal.SIGTERM))
    process.wait()

def run_step1_single_job(mode, partition, job_name, num_gpus, output_log, method_name, experiment_name, behaviors_path, save_dir,
                         start_idx, end_idx, behavior_ids_subset, run_id, overwrite, verbose, seed):
    if mode == "slurm":
        overwrite_string = "True" if overwrite else "False"
        verbose_string = "True" if verbose else "False"
        behavior_ids_subset = '""' if not behavior_ids_subset else behavior_ids_subset
        run_id_string = '""' if not run_id else run_id
        # Construct the sbatch command with all parameters and script arguments
        command = f'''sbatch --partition={partition} --comment=pytorch --job-name={job_name} --nodes=1 --gpus-per-node={num_gpus} --time=24:00:00 --output={output_log} \
            scripts/generate_test_cases.sh {method_name} {experiment_name} {behaviors_path} {save_dir} {start_idx} {end_idx} {behavior_ids_subset} {run_id_string} {overwrite_string} {verbose_string}'''
        # Run the sbatch command and get the job ID
        job_id = run_subprocess_slurm(command)
        return job_id
    
    elif mode in ["local", "local_parallel"]:
        # Construct the local command with all parameters and script arguments (with -u in case the output is piped to another file)
        options = [f"--{opt}" for opt, val in zip(['overwrite', 'verbose'], [overwrite, verbose]) if val]
        command = f"python -u generate_test_cases.py --method_name={method_name} --experiment_name={experiment_name} --behaviors_path={behaviors_path} \
            --save_dir={save_dir} --behavior_start_idx={start_idx} --behavior_end_idx={end_idx} --behavior_ids_subset={behavior_ids_subset} --run_id={run_id} --seed={seed} " + ' '.join(options)
        # Execute the local command
        if mode == "local":
            print(command)
            subprocess.run(command, shell=True, capture_output=False, text=True)
        # Return command to be executed parallel
        elif mode == "local_parallel":
            future = run_subprocess_ray.options(resources={"visible_gpus": num_gpus}).remote(command)
            return future

def run_step1_5_single_job(mode, partition, job_name, output_log, method_name, save_dir, seed, extract_mode, dependency_job_ids=None):
    if mode == "slurm":
        # Construct the sbatch command with all parameters and script arguments
        command = f"sbatch --partition={partition} --comment=python --job-name={job_name} --nodes=1 --gpus-per-node=0 --time=00:10:00 --output={output_log}"
        if dependency_job_ids:
            command += f" --dependency=afterok:{':'.join(dependency_job_ids)}"
        command += f" scripts/extract_test_cases.sh {method_name} {save_dir} {extract_mode}"
        # Run the sbatch command and get the job ID
        job_id = run_subprocess_slurm(command)
        return job_id
    elif mode in ["local", "local_parallel"]:
        # Construct the local command with all parameters and script arguments (with -u in case the output is piped to another file)
        command = f"python -u extract_all_test_cases.py --method_name={method_name} --save_dir={save_dir} --extract_mode={extract_mode} --seed={seed}"
        # Execute the local command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Return True to indicate job was executed (for local mode job counting)
        return "local_job_completed" if result.returncode == 0 else None


def run_step2_single_job(mode, partition, job_name, num_gpus, output_log, model_name, behaviors_path, test_cases_path, save_path,
                         max_new_tokens, incremental_update, batch_size, generate_with_vllm, seed, 
                         enable_max_length_regeneration=False, max_regeneration_tokens=1024, dependency_job_ids=None):
    if mode == "slurm":
        incremental_update_str = "True" if incremental_update else "False"
        generate_with_vllm_str = "True" if generate_with_vllm else "False"
        enable_max_regen_str = "True" if enable_max_length_regeneration else "False"
        
        # Construct the sbatch command with all parameters and script arguments
        command = f"sbatch --partition={partition} --comment=pytorch --job-name={job_name} --nodes=1 --gpus-per-node={num_gpus} --time=10:00:00 --output={output_log}"
        if dependency_job_ids:
            command += f" --dependency=afterok:{':'.join(dependency_job_ids)}"
        # Updated script call with additional parameters for max length regeneration
        command += f" scripts/generate_completions_modified.sh {model_name} {behaviors_path} {test_cases_path} {save_path} {max_new_tokens} {incremental_update_str} {batch_size} {generate_with_vllm_str} {enable_max_regen_str} {max_regeneration_tokens}"
        # Run the sbatch command and get the job ID
        job_id = run_subprocess_slurm(command)
        return job_id
    
    elif mode in ["local", "local_parallel"]:
        # Construct the local command with all parameters and script arguments (with -u in case the output is piped to another file)
        incremental_update_str = "--incremental_update" if incremental_update else ""
        generate_with_vllm_str = "--generate_with_vllm" if generate_with_vllm else ""
        enable_max_regen_str = "--enable_max_length_regeneration" if enable_max_length_regeneration else ""
        
        command = f"python -u generate_completions_modified.py --model_name={model_name} --behaviors_path={behaviors_path} --test_cases_path={test_cases_path} \
            --save_path={save_path} --max_new_tokens={max_new_tokens} --batch_size={batch_size} {incremental_update_str} {generate_with_vllm_str} \
            {enable_max_regen_str} --max_regeneration_tokens={max_regeneration_tokens} --seed={seed}"
        # Execute the local command
        if mode == 'local':
            print(f"Generating completions: {command}")
            result = subprocess.run(command, shell=True, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"Error in completion generation: {result.returncode}")
        else:
            future = run_subprocess_ray.options(resources={"visible_gpus": num_gpus}).remote(command)
            return future

def run_step3_single_job(mode, partition, job_name, output_log, cls_path, behaviors_path, completions_path, save_path,
                         use_advbench_first, include_advbench_metric, num_tokens, seed, save_first_success_only=False, dependency_job_ids=None):
    if mode == "slurm":
        use_advbench_first_str = "True" if use_advbench_first else "False"
        include_advbench_metric_str = "True" if include_advbench_metric else "False"
        save_first_success_only_str = "True" if save_first_success_only else "False"
        # Construct the sbatch command with all parameters and script arguments
        command = f"sbatch --partition={partition} --comment=pytorch --job-name={job_name} --nodes=1 --gpus-per-node=1 --time=5:00:00 --output={output_log}"
        if dependency_job_ids:
            command += f" --dependency=afterok:{':'.join(dependency_job_ids)}"
        # Updated script call with additional parameters including seed and save_first_success_only
        command += f" scripts/evaluate_completions_modified.sh {cls_path} {behaviors_path} {completions_path} {save_path} {use_advbench_first_str} {include_advbench_metric_str} {num_tokens} {seed} {save_first_success_only_str}"
        # Run the sbatch command and get the job ID
        job_id = run_subprocess_slurm(command)
        return job_id

    elif mode in ["local", "local_parallel"]:
        # Construct the local command with all parameters and script arguments (with -u in case the output is piped to another file)
        use_advbench_first_str = "--use_advbench_first" if use_advbench_first else ""
        include_advbench_metric_str = "--include_advbench_metric" if include_advbench_metric else ""
        save_first_success_only_str = "--save_first_success_only" if save_first_success_only else ""
        command = f"python -u evaluate_completions_modified_GPT.py --cls_path={cls_path} --behaviors_path={behaviors_path} --completions_path={completions_path} \
            --save_path={save_path} {use_advbench_first_str} {include_advbench_metric_str} --num_tokens={num_tokens} --seed={seed} {save_first_success_only_str}"

        # Execute the local command
        if mode == "local":
            print(f"Evaluating completions: {command}")
            result = subprocess.run(command, shell=True, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"Error in evaluation: {result.returncode}")
        # Return command to be executed parallel
        elif mode == "local_parallel":
            future = run_subprocess_ray.options(resources={"visible_gpus": 1}).remote(command)
            return future


def run_step1(args, expanded_pipeline_configs):
    # Compute num_behaviors to get end_idx
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)
    num_behaviors = len(behaviors)

    # Handling behavior_ids_subset
    if args.behavior_ids_subset:
        if os.path.exists(args.behavior_ids_subset):
            with open(args.behavior_ids_subset, 'r') as f:
                behavior_ids_subset = f.read().splitlines()
        else:
            behavior_ids_subset = args.behavior_ids_subset.split(',')
        
        behaviors = [b for b in behaviors if b['BehaviorID'] in behavior_ids_subset]
        num_behaviors = len(behaviors)

    print(f"Step 1: Generating test cases for {num_behaviors} behaviors")

    # Running jobs
    job_ids = {}
    num_job_ids = 0

    for expanded_pipeline_config in expanded_pipeline_configs.values():
        method_name = expanded_pipeline_config['class_name']
        if method_name not in job_ids:
            job_ids[method_name] = {}
        base_num_gpus = expanded_pipeline_config['base_num_gpus']

        for experiment in expanded_pipeline_config['experiments']:
            experiment_name = experiment['experiment_name']
            if experiment_name not in job_ids[method_name]:
                job_ids[method_name][experiment_name] = []
            num_gpus = base_num_gpus + experiment['experiment_num_gpus']
            save_dir=os.path.join(args.base_save_dir, method_name, experiment_name, 'test_cases')

            if experiment['behavior_chunk_size'] == 'all_behaviors':
                behavior_chunk_size = num_behaviors
            else:
                behavior_chunk_size = experiment['behavior_chunk_size']

            for start_idx in tqdm(range(0, num_behaviors, behavior_chunk_size), desc=f"Scheduling {method_name}_{experiment_name}"):
                end_idx = str(start_idx + behavior_chunk_size)
                start_idx = str(start_idx)
                if (behavior_ids_subset := args.behavior_ids_subset):
                    behavior_ids_subset = ','.join([b['BehaviorID'] for b in behaviors[int(start_idx):int(end_idx)]])
                
                # Create output directory
                os.makedirs(os.path.dirname(os.path.join(args.base_log_dir, 'generate_test_cases', method_name, experiment_name, f'start_idx_{start_idx}.log')), exist_ok=True)
                output_log = os.path.join(args.base_log_dir, 'generate_test_cases', method_name, experiment_name, f'start_idx_{start_idx}.log')
                run_ids = expanded_pipeline_config.get('run_ids', [''])

                for run_id in run_ids:
                    
                    if run_id != '':
                        job_name=f"generate_test_cases_{method_name}_{experiment_name}_{start_idx}_{end_idx}_{run_id}"
                    else:
                        job_name=f"generate_test_cases_{method_name}_{experiment_name}_{start_idx}_{end_idx}"

                    job_id = run_step1_single_job(args.mode, args.partition, job_name, num_gpus, output_log, method_name, experiment_name,
                                                  args.behaviors_path, save_dir, start_idx, end_idx, behavior_ids_subset, run_id,
                                                  args.overwrite, args.verbose, args.seed)
                    if job_id:
                        job_ids[method_name][experiment_name].append(job_id)
                        num_job_ids += 1

    print(f"Step 1: Scheduled {num_job_ids} test case generation jobs")
    
    if num_job_ids == 0:
        return None
    
    # ======== parallel with ray for local_paralllel (NOTE: experimental mode)=====
    if args.mode == "local_parallel":
        # flatten all jobs
        futures = [job_id for method in job_ids.values() for experiment in method.values() for job_id in experiment]
        ray.get(futures)
        return None
    
    return job_ids


def run_step1_5(args, expanded_pipeline_configs, dependency_job_ids=None):
    print(f"Step 1.5: Merging and then extracting test cases with mode '{args.extract_mode}'")
    job_ids = {}
    num_job_ids = 0

    for expanded_pipeline_config in expanded_pipeline_configs.values():
        method_name = expanded_pipeline_config['class_name']
        if method_name not in job_ids:
            job_ids[method_name] = {}

        for experiment in expanded_pipeline_config['experiments']:
            experiment_name = experiment['experiment_name']
            if experiment_name not in job_ids[method_name]:
                job_ids[method_name][experiment_name] = []
            save_dir=os.path.join(args.base_save_dir, method_name, experiment_name, 'test_cases')

            # First, merge test cases
            print(f"  Merging test cases for {method_name}/{experiment_name}")
            merge_command = f"python -u merge_test_cases.py --method_name={method_name} --save_dir={save_dir} --seed={args.seed}"
            subprocess.run(merge_command, shell=True)

            # Then, extract test cases
            job_name=f"extract_test_cases_{method_name}_{experiment_name}"

            # Create output directory
            os.makedirs(os.path.dirname(os.path.join(args.base_log_dir, 'extract_test_cases', method_name, f'{experiment_name}.log')), exist_ok=True)
            output_log = os.path.join(args.base_log_dir, 'extract_test_cases', method_name, f'{experiment_name}.log')

            if dependency_job_ids and method_name in dependency_job_ids and experiment_name in dependency_job_ids[method_name]:
                current_dep_job_ids = dependency_job_ids[method_name][experiment_name]
            else:
                current_dep_job_ids = None

            job_id = run_step1_5_single_job(args.mode, args.partition, job_name, output_log, method_name, save_dir, args.seed,
                                            args.extract_mode, dependency_job_ids=current_dep_job_ids)
            if job_id:
                job_ids[method_name][experiment_name].append(job_id)
                num_job_ids += 1

    # Print appropriate message based on mode
    if args.mode == "slurm":
        print(f"Step 1.5: Completed merge and scheduled {num_job_ids} test case extraction jobs")
    else:
        print(f"Step 1.5: Completed merge and extracted test cases for {num_job_ids} experiments")

    if num_job_ids == 0:
        job_ids = None
    return job_ids


def run_step2(args, expanded_pipeline_configs, model_configs, dependency_job_ids=None):
    regeneration_info = ""
    if args.enable_max_length_regeneration:
        regeneration_info = f" with max length regeneration ({args.max_regeneration_tokens} tokens)"
    
    print(f"Step 2: Generating completions (max_tokens={args.max_new_tokens}, batch_size={args.batch_size}){regeneration_info}")
    job_ids = {}
    num_job_ids = 0

    for expanded_pipeline_config in expanded_pipeline_configs.values():
        method_name = expanded_pipeline_config['class_name']
        if method_name not in job_ids:
            job_ids[method_name] = {}

        for experiment in expanded_pipeline_config['experiments']:
            experiment_name = experiment['experiment_name']
            if experiment_name not in job_ids[method_name]:
                job_ids[method_name][experiment_name] = []

            for model_name in experiment['expected_target_models']:
                num_gpus = model_configs[model_name].get('num_gpus', 0)
                test_cases_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'test_cases', 'test_cases.json')
                
                # Check if test_cases.json exists
                if not os.path.exists(test_cases_path):
                    print(f"Warning: Test cases file not found: {test_cases_path}")
                    continue
                
                if args.max_new_tokens != 512:
                    save_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'completions',
                                             f'{args.max_new_tokens}_tokens', f'{model_name}.json')
                else:
                    save_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'completions', f'{model_name}.json')

                # Create output directory
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(args.base_log_dir, 'generate_completions', method_name, experiment_name, f'{model_name}.log')), exist_ok=True)
                
                job_name=f"generate_completions_{method_name}_{experiment_name}_{model_name}"
                output_log = os.path.join(args.base_log_dir, 'generate_completions', method_name, experiment_name, f'{model_name}.log')

                if dependency_job_ids and method_name in dependency_job_ids and experiment_name in dependency_job_ids[method_name]:
                    current_dep_job_ids = dependency_job_ids[method_name][experiment_name]
                else:
                    current_dep_job_ids = None
                    
                job_id = run_step2_single_job(args.mode, args.partition, job_name, num_gpus, output_log, model_name, args.behaviors_path,
                                              test_cases_path, save_path, args.max_new_tokens, args.incremental_update, 
                                              args.batch_size, args.generate_with_vllm, args.seed,
                                              args.enable_max_length_regeneration, args.max_regeneration_tokens,
                                              dependency_job_ids=current_dep_job_ids)
                if job_id:
                    job_ids[method_name][experiment_name].append(job_id)
                    num_job_ids += 1

    print(f"Step 2: Scheduled {num_job_ids} completion generation jobs")
    
    if num_job_ids == 0:
        job_ids = None

    if args.mode == "local_parallel":
        # flatten all jobs
        futures = [job_id for method in job_ids.values() for experiment in method.values() for job_id in experiment]
        ray.get(futures)
        return None
    
    return job_ids


def run_step3(args, expanded_pipeline_configs, dependency_job_ids=None):
    print(f"Step 3: Evaluating completions (advbench_first={args.use_advbench_first}, num_tokens={args.num_tokens})")
    job_ids = {}
    num_job_ids = 0

    for expanded_pipeline_config in expanded_pipeline_configs.values():
        method_name = expanded_pipeline_config['class_name']
        if method_name not in job_ids:
            job_ids[method_name] = {}

        for experiment in expanded_pipeline_config['experiments']:
            experiment_name = experiment['experiment_name']
            if experiment_name not in job_ids[method_name]:
                job_ids[method_name][experiment_name] = []

            for model_name in experiment['expected_target_models']:
                if args.max_new_tokens != 512:
                    completions_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'completions',
                                                    f'{args.max_new_tokens}_tokens', f'{model_name}.json')
                    save_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'results',
                                             f'{args.max_new_tokens}_tokens', f'{model_name}.json')
                else:
                    completions_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'completions', f'{model_name}.json')
                    save_path = os.path.join(args.base_save_dir, method_name, experiment_name, 'results', f'{model_name}.json')

                # Check if completions file exists
                if not os.path.exists(completions_path):
                    print(f"Warning: Completions file not found: {completions_path}")
                    continue

                # Create output directories
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                os.makedirs(os.path.dirname(os.path.join(args.base_log_dir, 'evaluate_completions', method_name, experiment_name, f'{model_name}.log')), exist_ok=True)

                job_name=f"evaluate_completions_{method_name}_{experiment_name}_{model_name}"
                output_log = os.path.join(args.base_log_dir, 'evaluate_completions', method_name, experiment_name, f'{model_name}.log')

                if dependency_job_ids and method_name in dependency_job_ids and experiment_name in dependency_job_ids[method_name]:
                    current_dep_job_ids = dependency_job_ids[method_name][experiment_name]
                else:
                    current_dep_job_ids = None
                    
                job_id = run_step3_single_job(args.mode, args.partition, job_name, output_log, args.cls_path, args.behaviors_path,
                                     completions_path, save_path, args.use_advbench_first, args.include_advbench_metric, 
                                     args.num_tokens, args.seed, args.save_first_success_only, dependency_job_ids=current_dep_job_ids)
                if job_id:
                    job_ids[method_name][experiment_name].append(job_id)
                    num_job_ids += 1
    
    print(f"Step 3: Scheduled {num_job_ids} evaluation jobs")

    if args.mode == "local_parallel":
        # flatten all jobs
        futures = [job_id for method in job_ids.values() for experiment in method.values() for job_id in experiment]
        ray.get(futures)

def create_shell_scripts():
    """
    Create shell scripts for the modified pipeline steps if they don't exist
    """
    scripts_dir = "scripts"
    os.makedirs(scripts_dir, exist_ok=True)

    # Create extract_test_cases.sh
    extract_script = """#!/bin/bash

METHOD_NAME=$1
SAVE_DIR=$2
EXTRACT_MODE=$3

python extract_all_test_cases.py \\
    --logs_file="$SAVE_DIR/logs.json" \\
    --output_file="$SAVE_DIR/test_cases.json" \\
    --extract_mode="$EXTRACT_MODE"
"""
    
    with open(os.path.join(scripts_dir, "extract_test_cases.sh"), "w") as f:
        f.write(extract_script)
    os.chmod(os.path.join(scripts_dir, "extract_test_cases.sh"), 0o755)

    # Create generate_completions_modified.sh (updated with max length regeneration support)
    completions_script = """#!/bin/bash

MODEL_NAME=$1
BEHAVIORS_PATH=$2
TEST_CASES_PATH=$3
SAVE_PATH=$4
MAX_NEW_TOKENS=$5
INCREMENTAL_UPDATE=$6
BATCH_SIZE=$7
GENERATE_WITH_VLLM=$8
ENABLE_MAX_REGEN=$9
MAX_REGENERATION_TOKENS=${10}

INCREMENTAL_FLAG=""
if [ "$INCREMENTAL_UPDATE" = "True" ]; then
    INCREMENTAL_FLAG="--incremental_update"
fi

VLLM_FLAG=""
if [ "$GENERATE_WITH_VLLM" = "True" ]; then
    VLLM_FLAG="--generate_with_vllm"
fi

MAX_REGEN_FLAG=""
if [ "$ENABLE_MAX_REGEN" = "True" ]; then
    MAX_REGEN_FLAG="--enable_max_length_regeneration"
fi

python generate_completions_modified.py \\
    --model_name="$MODEL_NAME" \\
    --behaviors_path="$BEHAVIORS_PATH" \\
    --test_cases_path="$TEST_CASES_PATH" \\
    --save_path="$SAVE_PATH" \\
    --max_new_tokens="$MAX_NEW_TOKENS" \\
    --batch_size="$BATCH_SIZE" \\
    --max_regeneration_tokens="$MAX_REGENERATION_TOKENS" \\
    $INCREMENTAL_FLAG \\
    $VLLM_FLAG \\
    $MAX_REGEN_FLAG
"""
    
    with open(os.path.join(scripts_dir, "generate_completions_modified.sh"), "w") as f:
        f.write(completions_script)
    os.chmod(os.path.join(scripts_dir, "generate_completions_modified.sh"), 0o755)

    # Create evaluate_completions_modified.sh
    evaluation_script = """#!/bin/bash

CLS_PATH=$1
BEHAVIORS_PATH=$2
COMPLETIONS_PATH=$3
SAVE_PATH=$4
USE_ADVBENCH_FIRST=$5
INCLUDE_ADVBENCH_METRIC=$6
NUM_TOKENS=$7
SEED=$8
SAVE_FIRST_SUCCESS_ONLY=$9

ADVBENCH_FIRST_FLAG=""
if [ "$USE_ADVBENCH_FIRST" = "True" ]; then
    ADVBENCH_FIRST_FLAG="--use_advbench_first"
fi

ADVBENCH_METRIC_FLAG=""
if [ "$INCLUDE_ADVBENCH_METRIC" = "True" ]; then
    ADVBENCH_METRIC_FLAG="--include_advbench_metric"
fi

SAVE_FIRST_SUCCESS_FLAG=""
if [ "$SAVE_FIRST_SUCCESS_ONLY" = "True" ]; then
    SAVE_FIRST_SUCCESS_FLAG="--save_first_success_only"
fi

python evaluate_completions_modified_GPT.py \\
    --cls_path="$CLS_PATH" \\
    --behaviors_path="$BEHAVIORS_PATH" \\
    --completions_path="$COMPLETIONS_PATH" \\
    --save_path="$SAVE_PATH" \\
    --num_tokens="$NUM_TOKENS" \\
    --seed="$SEED" \\
    $ADVBENCH_FIRST_FLAG \\
    $ADVBENCH_METRIC_FLAG \\
    $SAVE_FIRST_SUCCESS_FLAG
"""
    
    with open(os.path.join(scripts_dir, "evaluate_completions_modified.sh"), "w") as f:
        f.write(evaluation_script)
    os.chmod(os.path.join(scripts_dir, "evaluate_completions_modified.sh"), 0o755)

    print("Created shell scripts for modified pipeline")


def main():
    args = parse_args()

    # Set up GPU environment
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")
    else:
        print("Using all visible GPUs")

    # Create shell scripts if needed
    create_shell_scripts()

    # Load configurations
    with open(args.pipeline_config_path) as file:
        pipeline_config = yaml.full_load(file)
    
    with open(args.models_config_path) as file:
        model_configs = yaml.full_load(file)
    
    # Get the pipeline config for each method
    if args.methods == "all":
        methods = list(pipeline_config.keys())
    else:
        methods = args.methods.split(',')
        for method in methods:
            assert method in pipeline_config, f"Method {method} not found in pipeline_config.yaml"
    
    if args.models == "all":
        models = list(model_configs.keys())
    else:
        models = args.models.split(',')
        for model in models:
            assert model in model_configs, f"Model {model} not found in models.yaml"

    expanded_pipeline_configs = get_expanded_pipeline_configs(pipeline_config, model_configs, methods, models)
    
    print(f"Pipeline Configuration:")
    print(f"  Methods: {methods}")
    print(f"  Models: {models}")
    print(f"  Step: {args.step}")
    print(f"  Mode: {args.mode}")
    print(f"  Base save dir: {args.base_save_dir}")
    print(f"  Extract mode: {args.extract_mode}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Max length regeneration: {args.enable_max_length_regeneration}")
    if args.enable_max_length_regeneration:
        print(f"  Max regeneration tokens: {args.max_regeneration_tokens}")
    print(f"  Use AdvBench first: {args.use_advbench_first}")
    
    # ===================== Run the pipeline ===================== #
    parallel_mode = args.mode == "local_parallel"
    if parallel_mode:
        print("Initializing Ray for parallel execution...")
        _init_ray(num_cpus=8, reinit=False, resources={"visible_gpus": torch.cuda.device_count()})
    
    try:
        if args.step == "1":
            run_step1(args, expanded_pipeline_configs)
        elif args.step == "1.5":
            run_step1_5(args, expanded_pipeline_configs)
        elif args.step == "2":
            run_step2(args, expanded_pipeline_configs, model_configs)
        elif args.step == "3":
            run_step3(args, expanded_pipeline_configs)
        elif args.step == "2_and_3":
            dependency_job_ids = run_step2(args, expanded_pipeline_configs, model_configs)
            run_step3(args, expanded_pipeline_configs, dependency_job_ids)
        elif args.step == "all":
            dependency_job_ids = run_step1(args, expanded_pipeline_configs)
            dependency_job_ids = run_step1_5(args, expanded_pipeline_configs, dependency_job_ids)
            dependency_job_ids = run_step2(args, expanded_pipeline_configs, model_configs, dependency_job_ids)
            run_step3(args, expanded_pipeline_configs, dependency_job_ids)
            
        print("Pipeline execution completed successfully!")
            
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise
    finally:
        if parallel_mode and ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()