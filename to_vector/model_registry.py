import json
from pathlib import Path

from spidr.models import SpidR
from transformers import HubertModel
from transformers import Wav2Vec2Model
from transformers import WavLMModel

SUPPORTED_MODEL_TYPES = (Wav2Vec2Model, WavLMModel, HubertModel, SpidR)
SUPPORTED_MODEL_TYPE_NAMES = ('wav2vec2', 'wavlm', 'hubert', 'spidr')


def is_supported_model(model):
    return isinstance(model, SUPPORTED_MODEL_TYPES)


def is_spidr_model(model):
    if isinstance(model, SpidR): return True
    model_name = type(model).__name__
    module_name = type(model).__module__
    return model_name == 'SpidR' or module_name.startswith('spidr')


def model_to_type(model):
    '''Get the specific model family for a resolved model instance.'''
    if is_spidr_model(model): return 'spidr'
    if 'Wav2Vec2ForPreTraining' in type(model).__name__:
        return 'wav2vec2-pretraining'
    if is_supported_model(model):
        model_name = type(model).__name__
        if 'Wav2Vec2' in model_name: return 'wav2vec2'
        if 'Hubert' in model_name: return 'hubert'
        if 'WavLM' in model_name: return 'wavlm'
    return 'unknown'

def filename_model_type(model_name_or_path=None, config_filename=None):
    '''Infer model type from a local config.json file when available.'''
    config_filename = infer_config_filename(model_name_or_path,
        config_filename=config_filename)
    config_dict = load_config_dict(config_filename)
    if config_dict is None: return None
    if 'model_type' in config_dict: return config_dict['model_type']
    if 'run' in config_dict and 'model_type' in config_dict['run']:
        return config_dict['run']['model_type']
    return None


def infer_config_filename(model_name_or_path=None, config_filename=None):
    '''Infer the config filename for a local model path.'''
    if config_filename is not None: return Path(config_filename)
    if model_name_or_path is None: return None
    path = Path(model_name_or_path)
    if path.is_dir(): return path / 'config.json'
    if path.is_file(): return path.parent / 'config.json'
    return None


def load_config_dict(config_filename):
    '''Load a json config file into a dictionary.'''
    if config_filename is None: return None
    config_filename = Path(config_filename)
    if not config_filename.exists(): return None
    with config_filename.open() as handle:
        return json.load(handle)
