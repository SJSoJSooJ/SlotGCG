# baselines
import importlib
from .model_utils import get_template, load_model_and_tokenizer, load_vllm_model, _init_ray

_method_mapping = {
    "GCG": "baselines.gcg",
    "GCG_posinit_attention": "baselines.gcg_posinit_attention",
    "GCG_hij": "baselines.gcg_hij",
    "GCG_hij_posinit_attention": "baselines.gcg_hij_posinit_attention",
    "GCG_posinit_random": "baselines.gcg_posinit_random",
    "I_GCG": "baselines.i_gcg",
    "I_GCG_posinit_attention": "baselines.i_gcg_posinit_attention",
    "AttnGCG" : "baselines.attngcg",
    "AttnGCG_posinit_attention" : "baselines.attngcg_posinit_attention",
}


def get_method_class(method):
    if method not in _method_mapping:
        raise ValueError(f"Can not find method {method}")
    module_path = _method_mapping[method]
    module = importlib.import_module(module_path)
    method_class = getattr(module, method)
    return method_class


def init_method(method_class, method_config):
    if method_class.use_ray:
        _init_ray(num_cpus=8)
    output = method_class(**method_config)
    return output

