import json
import os
from pathlib import Path
import warnings

from decouple import config
from huggingface_hub import login
import torch
from transformers import AutoFeatureExtractor
from transformers import AutoModel
from transformers import AutoModelForPreTraining
from transformers import AutoProcessor

from spidr.config import SpidRConfig
from spidr.models import SpidR

from .audio import load_audio
from . import model_registry


default_cache_directory = config('HF_HOME', default='~/.cache/huggingface')
if default_cache_directory.startswith('~'):
    default_cache_directory = os.path.expanduser(default_cache_directory)

wav2vec2_base = 'facebook/wav2vec2-base'
hubert_base = 'facebook/hubert-base-ls960'
wavlm_base = 'microsoft/wavlm-base-plus'
default_checkpoint = wav2vec2_base


def load_model(model_name_or_path=None, cache_directory=None, gpu=False,
    config_filename=None, strict=True):
    '''Load a model and route to the correct backend by model_type.'''
    if not model_name_or_path: model_name_or_path = default_checkpoint
    model_type = infer_model_type(model_name_or_path,
        config_filename=config_filename)
    if model_type == 'spidr':
        return load_spidr_model(model_name_or_path, gpu=gpu,
            config_filename=config_filename, strict=strict)
    return load_huggingface_model(model_name_or_path,
        cache_directory=cache_directory, gpu=gpu)


def load_huggingface_model(model_name_or_path=None, cache_directory=None,
    gpu=False):
    '''Load a Hugging Face speech model.'''
    if not model_name_or_path: model_name_or_path = default_checkpoint
    if not cache_directory: cache_directory = default_cache_directory
    model = AutoModel.from_pretrained(model_name_or_path,
        cache_dir=cache_directory)
    if gpu: model = move_model_to_gpu(model)
    return model


def load_spidr_model(checkpoint, gpu=False, config_filename=None,
    strict=True):
    '''Load a SpidR model from a local checkpoint.'''
    if checkpoint is None:
        raise ValueError('checkpoint is required for SpidR models')
    cfg = load_spidr_config(checkpoint, config_filename=config_filename)
    model = SpidR(cfg)
    checkpoint = Path(checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location='cpu',
        weights_only=True)
    if 'model' in checkpoint_data: state_dict = checkpoint_data['model']
    else: state_dict = checkpoint_data
    model.load_state_dict(state_dict, strict=strict)
    model.name_or_path = str(checkpoint)
    if gpu: model = move_model_to_gpu(model)
    return model


def load_model_pt(checkpoint=None, gpu=False):
    '''Load a Hugging Face pretraining model.'''
    if not checkpoint: checkpoint = default_checkpoint
    model_pt = AutoModelForPreTraining.from_pretrained(checkpoint)
    if gpu: model_pt = move_model_to_gpu(model_pt)
    return model_pt


def load_model_for_attention_extraction(model_name_or_path=None,
    cache_directory=None, gpu=False, config_filename=None, strict=True):
    '''Load a model for attention extraction.'''
    if not model_name_or_path: model_name_or_path = default_checkpoint
    model_type = infer_model_type(model_name_or_path,
        config_filename=config_filename)
    if model_type == 'spidr':
        return load_spidr_model(model_name_or_path, gpu=gpu,
            config_filename=config_filename, strict=strict)
    if not cache_directory: cache_directory = default_cache_directory
    model = AutoModel.from_pretrained(model_name_or_path,
        cache_dir=cache_directory, attn_implementation='eager')
    if gpu: model = move_model_to_gpu(model)
    return model


def load_hubert_base_model(cache_directory=None, gpu=False):
    '''Load the default HuBERT base model.'''
    return load_huggingface_model(hubert_base, cache_directory, gpu)


def load_wav2vec2_base_model(cache_directory=None, gpu=False):
    '''Load the default Wav2Vec2 base model.'''
    return load_huggingface_model(wav2vec2_base, cache_directory, gpu)


def load_processor(model_name_or_path=None):
    '''Load a processor.
    model_name_or_path: Hugging Face repo id or local path
    '''
    if model_name_or_path is None: model_name_or_path = default_checkpoint
    return AutoProcessor.from_pretrained(model_name_or_path)


def load_feature_extractor(model_name_or_path=None):
    '''Load feature extractor.
    model_name_or_path: Hugging Face repo id or local path
    '''
    if model_name_or_path is None: model_name_or_path = default_checkpoint
    return AutoFeatureExtractor.from_pretrained(model_name_or_path)


def prepare_model(model, gpu, cache_directory=None,
    for_attention_extraction=False, config_filename=None, strict=True):
    '''Prepare a model from an instance, local path, repo id, or None.'''
    if for_attention_extraction: loader = load_model_for_attention_extraction
    else: loader = load_model
    model_name = None
    if isinstance(model, (str, Path)): model_name = str(model)
    if model is None:
        model = loader(gpu=gpu, cache_directory=cache_directory,
            config_filename=config_filename, strict=strict)
    elif model_name is not None:
        model = loader(model_name, gpu=gpu,
            cache_directory=cache_directory,
            config_filename=config_filename, strict=strict)
    model_type = get_model_type(model)
    if model_type == 'unknown':
        supported_types = '\n'.join(['\t' + str(t)
            for t in model_registry.SUPPORTED_MODEL_TYPES])
        message = f'WARNING: Model {type(model)} may not be supported. '
        message += 'Make sure it is one of the supported models: \n'
        message += supported_types
        print(message)
    if for_attention_extraction and model_type != 'spidr':
        if model.config._attn_implementation != 'eager':
            message = f'WARNING: Model {model.base_model_prefix} '
            message += 'is not configured for eager attention extraction. '
            message += 'Attention extraction may not work as expected.'
            print(message)
    return model


def prepare_feature_extractor(model):
    '''Prepare a feature extractor for Hugging Face-style models.'''
    if get_model_type(model) == 'spidr': return None
    model_name_or_path = getattr(model, 'name_or_path', None)
    if model_name_or_path:
        model_path = Path(model_name_or_path)
        if model_path.exists() or '/' in model_name_or_path:
            return load_feature_extractor(model_name_or_path)
    return load_feature_extractor()


def infer_model_type(model_name_or_path=None, config_filename=None):
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


def spidr_config_to_kwargs(config_dict):
    '''Extract SpidR config kwargs from a json dictionary.'''
    if config_dict is None: return {}
    if 'model' in config_dict and isinstance(config_dict['model'], dict):
        return config_dict['model']
    if 'model_type' in config_dict:
        return {key: value for key, value in config_dict.items()
            if key != 'model_type'}
    return config_dict


def load_spidr_config(checkpoint, config_filename=None):
    '''Load SpidR config from file or fall back to default SpidRConfig().'''
    config_filename = infer_config_filename(checkpoint,
        config_filename=config_filename)
    config_dict = load_config_dict(config_filename)
    if config_dict is None:
        warnings.warn(
            'No config.json was found for the SpidR checkpoint. '
            'Using default SpidRConfig().')
        return SpidRConfig()
    return SpidRConfig(**spidr_config_to_kwargs(config_dict))


def model_device(model):
    '''Get the device of the model.'''
    return next(model.parameters()).device


def model_is_on_gpu(model):
    '''Check if the model is on GPU.'''
    return model_device(model).type == 'cuda'


def model_is_on_cpu(model):
    '''Check if the model is on CPU.'''
    return model_device(model).type == 'cpu'


def model_is_spidr(model):
    '''Check whether a model is a SpidR model.'''
    if isinstance(model, SpidR): return True
    model_name = type(model).__name__
    module_name = type(model).__module__
    return model_name == 'SpidR' or module_name.startswith('spidr')


def login_huggingface(token=None):
    '''Login to Hugging Face if a token is configured or provided.'''
    if token is None: token = config('HF_TOKEN', default=None)
    if not token: return False
    login(token)
    return True


def get_model_type(model):
    '''Get the specific model family for a resolved model instance.'''
    if model_is_spidr(model): return 'spidr'
    if 'Wav2Vec2ForPreTraining' in type(model).__name__:
        return 'wav2vec2-pretraining'
    if model_registry.is_supported_model(model):
        model_name = type(model).__name__
        if 'Wav2Vec2' in model_name: return 'wav2vec2'
        if 'Hubert' in model_name: return 'hubert'
        if 'WavLM' in model_name: return 'wavlm'
    return 'unknown'


def move_model_to_gpu(model):
    '''Move the model to GPU if available.'''
    model_name = getattr(model, 'base_model_prefix', type(model).__name__)
    message = f'No GPU available. Model {model_name} will remain on CPU.'
    if model_is_on_gpu(model):
        print('Model is already on GPU.')
        return model
    if torch.cuda.is_available():
        print(f'Moving model {model_name} to GPU.')
        model.to('cuda')
    else:
        print('WARNING:', message)
    return model


def move_model_to_cpu(model):
    '''Move the model to CPU.'''
    model_name = getattr(model, 'base_model_prefix', type(model).__name__)
    if model_is_on_cpu(model):
        print('Model is already on CPU.')
        return model
    print(f'Moving model {model_name} to CPU.')
    model.to('cpu')
    return model


def move_model(model, gpu=False):
    '''Move the model to GPU if gpu is True, otherwise to CPU.'''
    if gpu: return move_model_to_gpu(model)
    return move_model_to_cpu(model)
