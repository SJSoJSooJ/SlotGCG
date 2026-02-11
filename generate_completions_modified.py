import transformers
import json
import argparse
import os
import csv
from tqdm import tqdm 
import torch
from baselines import get_template, load_model_and_tokenizer, load_vllm_model
from generate_utils import set_random_seed
from api_models import api_models_map
from functools import partial
import multimodalmodels
from PIL import Image
from torchvision import transforms
from vllm import SamplingParams
from accelerate.utils import find_executable_batch_size
from eval_utils import advbench_metric_test_prefixes
import yaml

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate completions for test cases.")
    parser.add_argument("--model_name", type=str,
                        help="The name of the model in the models config file")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--test_cases_path", type=str,
                        help="The path to the test cases file to generate completions for")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving completions")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for completions")
    parser.add_argument("--generate_with_vllm", action="store_true",
                        help="Whether to generate completions with vLLM (if applicable)")
    parser.add_argument("--incremental_update", action="store_true",
                        help="Whether to incrementally update completions or generate a new completions file from scratch")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation (will be auto-adjusted if OOM)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--enable_max_length_regeneration", action="store_true", default=False,
                        help="Enable max length regeneration for AdvBench successful cases")
    parser.add_argument("--max_regeneration_tokens", type=int, default=1024,
                        help="Maximum tokens for regeneration when AdvBench succeeds")
    args = parser.parse_args()
    return args


def check_advbench_success(generation):
    """Check if generation is successful according to AdvBench metric"""
    if not generation:
        return False
    generation = generation.strip()
    jailbroken = not any([prefix in generation for prefix in advbench_metric_test_prefixes])
    return jailbroken


def regenerate_with_max_tokens(generation_function, test_case, max_tokens):
    """Regenerate a single test case with maximum tokens"""
    try:
        regenerated = generation_function(test_cases=[test_case], max_new_tokens=max_tokens)
        return regenerated[0] if regenerated else None
    except Exception as e:
        print(f"Error during regeneration: {e}")
        return None


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    set_random_seed(args.seed)
    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)
    
    num_gpus = model_configs[args.model_name].get('num_gpus', 0)
    # check num gpus available to see if greater than num_gpus in config
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available != num_gpus:
        print(f"Warning: num_gpus in config ({num_gpus}) does not match num_gpus available ({num_gpus_available}). Using {num_gpus_available} GPUs.")
        num_gpus = num_gpus_available
    model_config = model_configs[args.model_name]['model']
    model_config['num_gpus'] = num_gpus

    print("model_config", model_config)
    
    # ========== load test cases ========== #
    # load behaviors csv for reference (optional)
    behaviors = {}
    if os.path.exists(args.behaviors_path):
        with open(args.behaviors_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for behavior in reader:
                behaviors[behavior['BehaviorID']] = behavior
        print(f"Loaded {len(behaviors)} behaviors from {args.behaviors_path}")
   
    # Load test cases
    with open(args.test_cases_path, 'r') as f:
        test_cases_data = json.load(f)
    print(f"Loaded test cases for {len(test_cases_data)} behavior categories")
    
    # Print statistics about test cases
    total_test_cases = sum(len(cases) for cases in test_cases_data.values())
    print(f"Total test cases to process: {total_test_cases}")
    for behavior_id, cases in test_cases_data.items():
        print(f"  {behavior_id}: {len(cases)} test cases")

    generation_function = load_generation_function(model_config,
                                                args.max_new_tokens,
                                                test_cases_path=args.test_cases_path,
                                                generate_with_vllm=args.generate_with_vllm,
                                                batch_size=args.batch_size)

    # Max length regeneration will reuse the main generation function
    if args.enable_max_length_regeneration:
        print(f"Max length regeneration enabled with {args.max_regeneration_tokens} tokens")

    returned_data = {}

    if args.incremental_update and os.path.exists(args.save_path):
        print(f"Loading existing completions from {args.save_path}")
        with open(args.save_path, 'r') as file:
            returned_data = json.load(file)
        
        # Remove test cases that have already been generated
        new_test_cases_data = {}
        for behavior_id, _test_cases in test_cases_data.items():
            # Check if behavior_id is in returned_data and completions have been generated for all test cases
            if behavior_id in returned_data and len(returned_data[behavior_id]) == len(_test_cases):
                print(f"Skipping {behavior_id}: completions already exist for all {len(_test_cases)} test cases")
                continue
            print(f"Processing {behavior_id}: {len(_test_cases)} test cases")
            new_test_cases_data[behavior_id] = _test_cases
        test_cases_data = new_test_cases_data

    # ==== Flatten dict to array ======
    test_cases = []
    for behavior_id, _test_cases in test_cases_data.items():
        test_cases.extend([dict(behavior_id=behavior_id, test_case=t) for t in _test_cases])

    if len(test_cases) == 0:
        print('No new test cases to generate completions for')
        return
    else:
        print(f'Generating completions for {len(test_cases)} test cases')

    # ==== Generate ====
    print('Generating completions...')
    generations = generation_function(test_cases=[t['test_case'] for t in test_cases])
    print('Generation completed')

    # ==== Check for AdvBench success and regenerate if enabled ====
    if args.enable_max_length_regeneration:
        generations = improved_regeneration_logic(
            test_cases, 
            generations, 
            generation_function,
            model_config,
            args
        )

    # ==== Unflatten into dict of {behavior_id: [{test_case: t, generation: g}, ..], ...}
    print('Organizing results...')
    for t, g in zip(test_cases, generations):
        returned_data.setdefault(t['behavior_id'], [])
        completion_entry = {
            'test_case': t['test_case'], 
            'generation': g
        }
        
        # Add regeneration metadata if applicable
        if args.enable_max_length_regeneration:
            completion_entry['advbench_success'] = check_advbench_success(g)
            completion_entry['max_tokens_used'] = args.max_regeneration_tokens if check_advbench_success(g) else args.max_new_tokens
        
        returned_data[t['behavior_id']].append(completion_entry)

    # Create directories for save_path
    print(f'Saving completions to {args.save_path}...')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    
    # Add metadata to the saved file
    save_data = returned_data
    if args.enable_max_length_regeneration:
        save_data = {
            'metadata': {
                'max_length_regeneration_enabled': True,
                'initial_max_tokens': args.max_new_tokens,
                'regeneration_max_tokens': args.max_regeneration_tokens,
                'advbench_success_count': sum(
                    sum(1 for completion in completions if completion.get('advbench_success', False))
                    for completions in returned_data.values()
                )
            },
            'completions': returned_data
        }
    
    with open(args.save_path, 'w') as file:
        json.dump(save_data, file, indent=4)
    
    # Print final statistics
    final_total = sum(len(cases) for cases in returned_data.values())
    print(f'Saved {final_total} total completions')
    for behavior_id, cases in returned_data.items():
        print(f"  {behavior_id}: {len(cases)} completions")
    
    if args.enable_max_length_regeneration:
        advbench_success_total = sum(
            sum(1 for completion in completions if completion.get('advbench_success', False))
            for completions in returned_data.values()
        )
        print(f'AdvBench successful cases: {advbench_success_total}/{final_total}')
    
    print('Done!')


def _vllm_generate(model, test_cases, template, **generation_kwargs):
    inputs = [template['prompt'].format(instruction=s) for s in test_cases]
    outputs = model.generate(inputs, **generation_kwargs)
    generations = [o.outputs[0].text.strip() for o in outputs]
    return generations

def _hf_generate_with_batching(model, tokenizer, test_cases, template, batch_size=8, **generation_kwargs):
    @find_executable_batch_size(starting_batch_size=min(batch_size, len(test_cases)))
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, template, generation_kwargs
        generations = []
        print(f"Using batch size: {batch_size}")
        for i in tqdm(range(0, len(test_cases), batch_size), desc="Generating batches"):
            batched_test_cases = test_cases[i:i+batch_size]
            inputs = [template['prompt'].format(instruction=s) for s in batched_test_cases]
            inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                outputs = model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'], **generation_kwargs).cpu()
            generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            batch_generations = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in generated_tokens]
            generations.extend(batch_generations)
        return generations
    return inner_generation_loop()

def _api_model_generate(model, test_cases, **generation_kwargs):
    # MultiModal test cases are in saved in pair of [[img, text], ...]
    if isinstance(test_cases[0], (tuple, list)):
        images, prompts = zip(*test_cases)
        return model.generate(prompts=list(prompts), images=list(images), **generation_kwargs)   
    return model.generate(prompts=test_cases, **generation_kwargs)

def load_generation_function(model_config, max_new_tokens, test_cases_path, generate_with_vllm=False, batch_size=8):
    model_name_or_path = model_config['model_name_or_path']

    if model_name_or_path in dir(multimodalmodels):
        model = getattr(multimodalmodels, model_name_or_path)()
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        image_dir = os.path.join(os.path.dirname(test_cases_path), 'images')
        return partial(model.generate, image_dir=image_dir, **generation_kwargs)
    if (model := api_models_map(**model_config)):
        generation_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.0, use_tqdm=True)
        return partial(_api_model_generate, model=model, **generation_kwargs)
    elif generate_with_vllm:
        print('Using vLLM generation')
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        model = load_vllm_model(**model_config)
        generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))
        return partial(_vllm_generate, model=model, template=TEMPLATE, **generation_kwargs)
    else:
        print('Using HF generation')
        model, tokenizer = load_model_and_tokenizer(**model_config)
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))
        return partial(_hf_generate_with_batching, model=model, tokenizer=tokenizer, template=TEMPLATE, batch_size=batch_size, **generation_kwargs)

def improved_regeneration_logic(test_cases, generations, generation_function, model_config, args):
    """
    Regeneration with model reloading to clear KV cache and prevent memory issues
    """
    print('Checking AdvBench success and regenerating successful cases with max tokens...')
    
    # Find successful cases
    successful_cases = []
    successful_indices = []
    
    for i, (test_case_info, generation) in enumerate(zip(test_cases, generations)):
        if check_advbench_success(generation):
            successful_cases.append(test_case_info['test_case'])
            successful_indices.append(i)
    
    if not successful_cases:
        print('No successful cases found for regeneration')
        return generations
    
    print(f'Found {len(successful_cases)} successful cases for regeneration')
    print('Clearing previous model and reloading for regeneration...')
    
    try:
        # ========== Clear existing model to free memory ========== #
        if args.generate_with_vllm:
            # Clear vLLM model
            if hasattr(generation_function, 'keywords') and 'model' in generation_function.keywords:
                old_model = generation_function.keywords['model']
                if hasattr(old_model, 'llm_engine'):
                    del old_model.llm_engine
                del old_model
                print('vLLM model cleared')
        elif not api_models_map(**model_config):  # HuggingFace models
            # Clear HuggingFace model and tokenizer
            if hasattr(generation_function, 'keywords'):
                if 'model' in generation_function.keywords:
                    old_model = generation_function.keywords['model']
                    if hasattr(old_model, 'cpu'):
                        old_model.cpu()
                    del old_model
                    print('HuggingFace model moved to CPU and cleared')
                if 'tokenizer' in generation_function.keywords:
                    del generation_function.keywords['tokenizer']
                    print('Tokenizer cleared')
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print('GPU cache cleared')
        
        # ========== Load new model for regeneration ========== #
        print('Loading new model for regeneration...')
        @find_executable_batch_size(starting_batch_size=min(1, len(successful_cases)))
        def load_and_regenerate(batch_size):
            print(f"Loading regeneration model with batch size: {batch_size}")
            
            regeneration_function = load_generation_function(
                model_config,
                args.max_regeneration_tokens,
                test_cases_path=args.test_cases_path,
                generate_with_vllm=args.generate_with_vllm,
                batch_size=batch_size
            )
            
            print('New model loaded successfully')
            print(f'Regenerating {len(successful_cases)} successful cases...')
            return regeneration_function(test_cases=successful_cases)
        print('New model loaded successfully')
        
        # ========== Perform regeneration ========== #
        print(f'Regenerating {len(successful_cases)} successful cases...')
        regenerated_outputs = load_and_regenerate()
        
        # Update the original generations
        regeneration_count = 0
        for idx, regen_output in zip(successful_indices, regenerated_outputs):
            if regen_output:
                generations[idx] = regen_output
                regeneration_count += 1
            else:
                print(f"Regeneration failed for test case at index {idx}, keeping original generation")
                
        print(f'Successfully regenerated {regeneration_count} cases with {args.max_regeneration_tokens} tokens')
        # Final GPU cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print('Regeneration model cleaned up')
        
    except Exception as e:
        print(f"Error during regeneration: {e}")
        print("Regeneration failed, keeping original generations")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return generations

if __name__ == "__main__":
    main()