# ============================== BASELINE TEMPLATE ============================== #
import time
from tqdm import tqdm
import numpy as np
import os
import json
from .model_utils import load_model_and_tokenizer, get_template
import transformers
import vllm
import ray
import fastchat

class RedTeamingMethod:
    use_ray = False
    default_dependencies = [transformers, vllm, ray, fastchat]
    def __init__(self):
        raise NotImplementedError

    def generate_test_cases(self, behaviors, verbose=False):
        raise NotImplementedError

    @staticmethod
    def get_output_file_path(save_dir, behavior_id, file_type, run_id=None):
        raise NotImplementedError

    def save_test_cases(self, save_dir, test_cases, logs=None, method_config=None, run_id=None):
        raise NotImplementedError

    @staticmethod
    def merge_test_cases(save_dir):
 
        pass
    
    def _replace_tokens(self, d):

        for k, v in d.items():
            if ('token' in k or 'apikey' in k.replace("_", "")) and isinstance(v, str):
                d[k] = '*' * (len(v) - 4) + v[-4:]
            elif isinstance(v, dict):
                self._replace_tokens(v)


class SingleBehaviorRedTeamingMethod(RedTeamingMethod):

    def __init__(self, target_model, num_test_cases_per_behavior=1, test_cases_batch_size=1, **kwargs):

        model_kwargs = target_model
        model, tokenizer = load_model_and_tokenizer(**model_kwargs)
        self.model_name_or_path = model_kwargs['model_name_or_path']
        self.model = model
        self.tokenizer = tokenizer
        self.num_test_cases_per_behavior = num_test_cases_per_behavior
        self.test_cases_batch_size = test_cases_batch_size

    def generate_test_cases(self, behaviors, verbose=False):

        # set self.test_cases_batch_size and self.num_test_cases_per_behavior to 1 if not defined yet
        # (enables subclassing with a new __init__ function that doesn't define these attributes)
        if not hasattr(self, 'test_cases_batch_size'):
            self.test_cases_batch_size = 1
        if not hasattr(self, 'num_test_cases_per_behavior'):
            self.num_test_cases_per_behavior = 1

        test_cases = {}
        logs = {}
        for behavior_dict in tqdm(behaviors, total=len(behaviors)):
            start_time = time.time()
            # break it down into batches
            num_batches = int(np.ceil(self.num_test_cases_per_behavior / self.test_cases_batch_size))
            remainder = self.num_test_cases_per_behavior % self.test_cases_batch_size
            current_test_cases = []
            current_logs = []
            for j in range(num_batches):
                # Calculate the current batch size
                if j == num_batches - 1 and remainder != 0:
                    current_batch_size = remainder
                else:
                    current_batch_size = self.test_cases_batch_size

                if verbose:
                    print(f"Generating batch {j+1}/{num_batches} with size {current_batch_size}")
                batch_test_cases, batch_logs = self.generate_test_cases_single_behavior(behavior_dict, num_generate=current_batch_size,
                                                                                        verbose=verbose)
                if isinstance(batch_test_cases, list) and (len(batch_test_cases) == len(batch_logs)):
                    current_test_cases.extend(batch_test_cases)
                    current_logs.extend(batch_logs)
                else:
                    current_test_cases.append(batch_test_cases)
                    current_logs.append(batch_logs)
            
            # add test cases and info to the output dictionaries
            behavior_id = behavior_dict['BehaviorID']

            test_cases[behavior_id] = current_test_cases
            logs[behavior_id] = current_logs

            if verbose:
                print(f"Time elapsed (s): {time.time() - start_time}")
        return test_cases, logs

    def generate_test_cases_single_behavior(self, behavior_dict, num_generate=1, verbose=False):

        raise NotImplementedError
    
    @staticmethod
    def get_output_file_path(save_dir, behavior_id, file_type, run_id=None):
        run_id = f"_{run_id}" if run_id else ""
        return os.path.join(save_dir, 'test_cases_individual_behaviors', behavior_id, f"{file_type}{run_id}.json")

    def save_test_cases(self, save_dir, test_cases, logs=None, method_config=None, run_id=None):
        for behavior_id in test_cases.keys():
            test_cases_individual_behavior = {behavior_id: test_cases[behavior_id]}
            if logs is not None:
                behavior_logs = {behavior_id: logs.get(behavior_id, [])}
            self.save_test_cases_single_behavior(save_dir, behavior_id, test_cases_individual_behavior,
                                                 behavior_logs, method_config=method_config, run_id=run_id)

    def save_test_cases_single_behavior(self, save_dir, behavior_id, test_cases, logs=None, method_config=None, run_id=None):
        if save_dir is None:
            print("save_dir is None")
            return
        
        test_cases_save_path = self.get_output_file_path(save_dir, behavior_id, 'test_cases', run_id)
        logs_save_path = self.get_output_file_path(save_dir, behavior_id, 'logs', run_id)
        method_config_save_path = self.get_output_file_path(save_dir, behavior_id, 'method_config', run_id)

        os.makedirs(os.path.dirname(test_cases_save_path), exist_ok=True)
        if test_cases is not None:
            with open(test_cases_save_path, 'w', encoding='utf-8') as f:
                json.dump(test_cases, f, indent=4)

        if logs is not None:
            with open(logs_save_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=4)

        if method_config is not None:
            # mask token or api_key
            self._replace_tokens(method_config)
            with open(method_config_save_path, 'w', encoding='utf-8') as f:
                method_config["dependencies"] = {l.__name__: l.__version__ for l in self.default_dependencies}
                json.dump(method_config, f, indent=4)

    @staticmethod
    def merge_test_cases(save_dir):

        test_cases = {}
        logs = {}
        assert os.path.exists(save_dir), f"{save_dir} does not exist"

        behavior_ids = [d for d in os.listdir(os.path.join(save_dir, 'test_cases_individual_behaviors'))]

        for behavior_id in behavior_ids:
            test_cases_path = os.path.join(save_dir, 'test_cases_individual_behaviors', behavior_id, 'test_cases.json')
            if os.path.exists(test_cases_path):
                with open(test_cases_path, 'r') as f:
                    test_cases_part = json.load(f)
                for behavior_id in test_cases_part:
                    assert behavior_id not in test_cases, f"Duplicate behavior ID: {behavior_id}"
                test_cases.update(test_cases_part)

        for behavior_id in behavior_ids:
            logs_path = os.path.join(save_dir, 'test_cases_individual_behaviors', behavior_id, 'logs.json')
            if os.path.exists(logs_path):
                with open(logs_path, 'r') as f:
                    logs_part = json.load(f)
                for behavior_id in logs_part:
                    assert behavior_id not in logs, f"Duplicate behavior ID: {behavior_id}"
                logs.update(logs_part)

        test_cases_save_path = os.path.join(save_dir, 'test_cases.json')
        logs_save_path = os.path.join(save_dir, 'logs.json')

        if len(test_cases) == 0:
            print(f"No test cases found in {save_dir}")
            return
        else:
            with open(test_cases_save_path, 'w') as f:
                json.dump(test_cases, f, indent=4)
            print(f"Saved test_cases.json to {test_cases_save_path}")
        
        if len(logs) == 0:
            print(f"No logs found in {save_dir}")
            return
        else:
            with open(logs_save_path, 'w') as f:
                json.dump(logs, f, indent=4)
            print(f"Saved logs.json to {logs_save_path}")
