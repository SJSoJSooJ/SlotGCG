import json
import itertools
from transformers import AutoTokenizer
import random
from defense.erase_and_check.defenses import delete_by_indices

def generate_erase_and_check_candidates(prompt, tokenizer, max_erase=20, mode="suffix", num_adv=2):
    """
    Generate all possible candidate prompts that would be checked by erase_and_check function.
    This mimics the logic from defenses.py but only generates candidates without evaluation.
    
    Args:
        prompt: The original prompt
        tokenizer: Tokenizer for the model
        max_erase: Maximum number of tokens to erase
        mode: "suffix", "infusion", or "insertion"
        num_adv: Number of adversarial prompts (for insertion mode)
    
    Returns:
        List of candidate prompts
    """
    
    if mode == "suffix":
        return generate_suffix_candidates(prompt, tokenizer, max_erase)
    elif mode == "infusion":
        return generate_infusion_candidates(prompt, tokenizer, max_erase)
    elif mode == "insertion":
        return generate_insertion_candidates(prompt, tokenizer, max_erase, num_adv)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def generate_suffix_candidates(prompt, tokenizer, max_erase=20):
    """Generate candidates by erasing tokens from the end (suffix mode)"""
    # Tokenize the prompt (skip the first token which is usually BOS)
    prompt_tokens = tokenizer(prompt)['input_ids'][1:]
    prompt_length = len(prompt_tokens)
    
    candidates = [prompt]  # Include original prompt
    
    # Erase tokens one by one from the end
    for i in range(min(max_erase, prompt_length)):
        erased_prompt_tokens = prompt_tokens[:-(i+1)]
        erased_prompt = tokenizer.decode(erased_prompt_tokens, skip_special_tokens=True)
        candidates.append(erased_prompt)
    
    return candidates

def generate_infusion_candidates(prompt, tokenizer, max_erase=3, max_candidates=20):
    """Generate candidates by sampling random subsets of tokens (infusion mode)"""
    import random
    from itertools import combinations
    
    # Tokenize the prompt
    prompt_tokens = tokenizer(prompt)['input_ids'][1:]
    prompt_length = len(prompt_tokens)
    
    candidates = [prompt]  # Include original prompt
    
    # Generate all possible combinations up to max_erase
    all_combinations = []
    for i in range(min(max_erase, prompt_length)):
        erase_locations = list(combinations(range(prompt_length), i+1))
        all_combinations.extend(erase_locations)
    
    # If we have too many combinations, sample randomly
    if len(all_combinations) > max_candidates - 1:  # -1 for original prompt
        sampled_combinations = random.sample(all_combinations, max_candidates - 1)
    else:
        sampled_combinations = all_combinations
    
    # Generate candidates from sampled combinations
    for location in sampled_combinations:
        erased_prompt_tokens = delete_by_indices(prompt_tokens, location)
        erased_prompt = tokenizer.decode(erased_prompt_tokens, skip_special_tokens=True)
        candidates.append(erased_prompt)
    
    return candidates

def generate_insertion_candidates(prompt, tokenizer, max_erase=5, num_adv=2):
    """Generate candidates for defending against multiple adversarial insertions"""
    # Tokenize the prompt
    prompt_tokens = tokenizer(prompt)['input_ids'][1:]
    prompt_length = len(prompt_tokens)
    
    candidates = {prompt}  # Use set to avoid duplicates
    
    # All possible gap and num_erase values
    args = []
    for k in range(num_adv):
        args.append(range(prompt_length))
        args.append(range(max_erase + 1))
    
    # Iterate over all possible combinations of gap and num_erase values
    for combination in itertools.product(*args):
        erase_locations = []
        start = 0
        end = 0
        
        for i in range(len(combination) // 2):
            start = end + combination[(2*i)]
            end = start + combination[(2*i) + 1]
            if start >= prompt_length or end > prompt_length:
                erase_locations = []
                break
            erase_locations.extend(range(start, end))
        
        if len(erase_locations) == 0 or len(erase_locations) > prompt_length:
            continue
        
        erased_prompt_tokens = delete_by_indices(prompt_tokens, erase_locations)
        erased_prompt = tokenizer.decode(erased_prompt_tokens, skip_special_tokens=True)
        candidates.add(erased_prompt)
    
    return list(candidates)

def generate_all_perturbed_inputs(test_cases_file, output_file, tokenizer_name="meta-llama/Llama-2-7b-hf", 
                                 max_erase=20, mode="suffix", num_adv=2, variations_per_prompt=None):
    """
    Generate perturbed inputs for all test cases and save in the format of all_perturbed_inputs.json
    
    Args:
        test_cases_file: Path to test_cases.json
        output_file: Path to save the perturbed inputs
        tokenizer_name: Name of the tokenizer to use
        max_erase: Maximum number of tokens to erase
        mode: Defense mode ("suffix", "infusion", or "insertion")
        num_adv: Number of adversarial prompts (for insertion mode)
        variations_per_prompt: Number of variations to generate per original prompt
    """
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load test cases
    print(f"Loading test cases from: {test_cases_file}")
    with open(test_cases_file, 'r') as f:
        test_cases = json.load(f)
    
    # Generate perturbed inputs
    all_perturbed = {}
    
    for test_id, prompts in test_cases.items():
        print(f"Processing {test_id}...")
        all_perturbed[test_id] = []
        
        for original_prompt in prompts:
            # Generate all possible candidates
            candidates = generate_erase_and_check_candidates(
                original_prompt, tokenizer, max_erase=max_erase, mode=mode, num_adv=num_adv
            )
            
            # Remove duplicates and empty strings
            candidates = list(set([c.strip() for c in candidates if c.strip()]))
            
            # If variations_per_prompt is None, use all candidates (no limit)
            if variations_per_prompt is None:
                selected = candidates
            else:
                # If we have more candidates than needed, sample randomly
                if len(candidates) > variations_per_prompt:
                    # Always include the original prompt
                    if original_prompt in candidates:
                        selected = [original_prompt]
                        remaining = [c for c in candidates if c != original_prompt]
                        selected.extend(random.sample(remaining, min(variations_per_prompt-1, len(remaining))))
                    else:
                        selected = random.sample(candidates, variations_per_prompt)
                else:
                    selected = candidates
            
            all_perturbed[test_id].extend(selected)
    
    # Save results
    print(f"Saving perturbed inputs to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_perturbed, f, indent=2)
    
    # Print statistics
    total_original = sum(len(prompts) for prompts in test_cases.values())
    total_perturbed = sum(len(prompts) for prompts in all_perturbed.values())
    
    print(f"\nStatistics:")
    print(f"Total original prompts: {total_original}")
    print(f"Total perturbed prompts: {total_perturbed}")
    print(f"Average variations per original: {total_perturbed/total_original:.1f}")
    print(f"Test cases processed: {len(all_perturbed)}")

if __name__ == "__main__":
    # Example usage
    generate_all_perturbed_inputs(
        test_cases_file="test_cases.json",
        output_file="generated_perturbed_inputs.json",
        tokenizer_name="meta-llama/Llama-2-7b-hf",
        max_erase=20,
        mode="suffix",  # Change to "infusion" or "insertion" as needed
        num_adv=2,
        variations_per_prompt=10
    )
    
    print("\nTo run with different modes:")
    print("- Suffix mode: python script.py --mode suffix --max_erase 20")
    print("- Infusion mode: python script.py --mode infusion --max_erase 2") 
    print("- Insertion mode: python script.py --mode insertion --max_erase 5 --num_adv 2")