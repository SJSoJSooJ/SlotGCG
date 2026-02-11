import json
import argparse
import os
from generate_utils import set_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Extract all test cases from logs.json to create test_cases.json.")

    # Method 1: Use method_name and save_dir (used by modified_run_pipeline.py)
    parser.add_argument("--method_name", type=str, default=None,
                        help="Method name (e.g., 'GCG')")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Save directory containing logs.json")

    # Method 2: Use explicit file paths (backward compatibility)
    parser.add_argument("--logs_file", type=str, default=None,
                        help="Path to the logs.json file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output path for the test_cases.json file")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--extract_mode", type=str, default='all',
                        choices=['all', 'last', 'best'],
                        help="Mode for extracting test cases: 'all' for all test cases, 'last' for final test case only, 'best' for lowest loss test case")
    args = parser.parse_args()

    # Determine logs_file and output_file based on which arguments were provided
    if args.save_dir is not None:
        # If save_dir is provided, construct the paths
        args.logs_file = os.path.join(args.save_dir, 'logs.json')
        args.output_file = os.path.join(args.save_dir, 'test_cases.json')
    elif args.logs_file is None:
        # If neither save_dir nor logs_file is provided, use defaults
        args.logs_file = 'logs.json'
        args.output_file = 'test_cases.json' if args.output_file is None else args.output_file
    elif args.output_file is None:
        # If logs_file is provided but output_file is not, use default
        args.output_file = 'test_cases.json'

    return args

def extract_test_cases(logs_file, output_file, extract_mode='all'):
    """
    Extract test cases from logs.json file
    
    Args:
        logs_file: Path to logs.json file
        output_file: Path to output test_cases.json file
        extract_mode: 'all', 'last', or 'best'
    """
    
    # Load logs.json
    with open(logs_file, 'r', encoding='utf-8') as f:
        logs_data = json.load(f)
    
    # Check if any behavior has 'all_test_cases'
    has_all_test_cases = False
    for key, value in logs_data.items():
        if isinstance(value, list) and len(value) > 0:
            experiment_data = value[0]
            if 'all_test_cases' in experiment_data:
                has_all_test_cases = True
                break
    
    if not has_all_test_cases:
        print(f"No 'all_test_cases' found in any behavior. Skipping extraction.")
        return
    
    test_cases_data = {}
    
    for key, value in logs_data.items():
        if isinstance(value, list) and len(value) > 0:
            # Each entry is a list containing experiment data
            experiment_data = value[0]  # Take the first (and likely only) experiment
            
            # Skip if 'all_test_cases' not found
            if 'all_test_cases' not in experiment_data:
                continue
            
            if extract_mode == 'all':
                # Extract all test cases
                test_cases_data[key] = experiment_data['all_test_cases']
                    
            elif extract_mode == 'last':
                # Extract only the last test case
                if experiment_data['all_test_cases']:
                    test_cases_data[key] = [experiment_data['all_test_cases'][-1]]
                    
            elif extract_mode == 'best':
                # Extract test case with the lowest loss
                if ('all_losses' in experiment_data and 
                    experiment_data['all_test_cases'] and 
                    experiment_data['all_losses']):
                    
                    min_loss_idx = experiment_data['all_losses'].index(min(experiment_data['all_losses']))
                    test_cases_data[key] = [experiment_data['all_test_cases'][min_loss_idx]]
    
    # Save to test_cases.json
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases_data, f, indent=4, ensure_ascii=False)
    
    # Print statistics
    total_test_cases = sum(len(cases) for cases in test_cases_data.values())
    print(f"Extraction completed!")
    print(f"Mode: {extract_mode}")
    print(f"Total categories: {len(test_cases_data)}")
    print(f"Total test cases: {total_test_cases}")
    print(f"Output saved to: {output_file}")
    
    # Print breakdown by category
    print("\nBreakdown by category:")
    for key, cases in test_cases_data.items():
        print(f"  {key}: {len(cases)} test cases")

def main():
    args = parse_args()
    print(args)
    set_random_seed(args.seed)
    
    if not os.path.exists(args.logs_file):
        print(f"Error: logs file '{args.logs_file}' not found!")
        return
    
    print(f'Extracting test cases from {args.logs_file}')
    extract_test_cases(args.logs_file, args.output_file, args.extract_mode)

if __name__ == "__main__":
    main()
