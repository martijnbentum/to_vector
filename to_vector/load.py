from decouple import config
from huggingface_hub import login
import librosa
import os
from pathlib import Path
import torch
from transformers import AutoFeatureExtractor
from transformers import AutoProcessor
from transformers import AutoModel
from transformers import AutoModelForPreTraining
from . import model_registry

default_cache_directory = config('HF_HOME', default='~/.cache/huggingface')
if default_cache_directory.startswith('~'):
    default_cache_directory = os.path.expanduser(default_cache_directory)

wav2vec2_base= 'facebook/wav2vec2-base'
hubert_base = 'facebook/hubert-base-ls960'
wavlm_base = 'microsoft/wavlm-base-plus'
default_checkpoint = wav2vec2_base


def login_huggingface(token=None):
    '''Login to Hugging Face if a token is configured or provided.'''
    if token is None: token = config('HF_TOKEN', default=None)
    if not token: return False
    login(token)
    return True

def load_audio(filename, start = 0.0, end=None):
    if end is None:
        duration = None
    else:
        if end < start:
            raise ValueError('end must be greater than or equal to start')
        duration = end - start
    audio, sr = librosa.load(filename, sr = 16000, offset=start,
        duration=duration)
    return audio

def load_processor(model_name_or_path = None):
    '''Load a processor. 
    model_name_or_path      Hugging Face repo id or local path
    '''
    if model_name_or_path is None: model_name_or_path = default_checkpoint
    return AutoProcessor.from_pretrained(model_name_or_path)

def load_feature_extractor(model_name_or_path = None):
    '''Load feature extractor.
    model_name_or_path      Hugging Face repo id or local path
    '''
    if model_name_or_path is None: model_name_or_path = default_checkpoint
    return AutoFeatureExtractor.from_pretrained(model_name_or_path) 

def load_pretrained_model(model_name_or_path = None, cache_directory = None, 
        gpu = False):
    if not model_name_or_path: model_name_or_path = default_checkpoint
    if not cache_directory: cache_directory = default_cache_directory
    model = AutoModel.from_pretrained(model_name_or_path,
        cache_dir = cache_directory)
    if gpu: model = move_model_to_gpu(model)
    return model

def load_model_pt(checkpoint = None, gpu = False):
    if not checkpoint: checkpoint = default_checkpoint
    model_pt = AutoModelForPreTraining.from_pretrained(checkpoint)
    if gpu: model_pt = move_model_to_gpu(model_pt)
    return model_pt

def load_model_for_attention_extraction(model_name_or_path = None, 
    cache_directory = None, gpu = False):
    '''Load a model for attention extraction. 
    This will load the model and move it to GPU if available and requested.
    model_name_or_path      Hugging Face repo id or local path
    cache_directory         Directory to cache the model
    gpu                     If True, move the model to GPU if available.
    '''
    if not model_name_or_path: model_name_or_path = default_checkpoint
    if not cache_directory: cache_directory = default_cache_directory
    model = AutoModel.from_pretrained(model_name_or_path, 
        cache_dir = cache_directory, attn_implementation = 'eager')
    if gpu: model = move_model_to_gpu(model)
    return model

def load_hubert_base_model(cache_directory = None, gpu = False):
    return load_pretrained_model(hubert_base, cache_directory, gpu)

def load_wav2vec2_base_model(cache_directory = None, gpu = False):
    return load_pretrained_model(wav2vec2_base, cache_directory, gpu)

def model_device(model):
    '''Get the device of the model.'''
    return next(model.parameters()).device

def model_is_on_gpu(model):
    '''Check if the model is on GPU.'''
    return model_device(model).type == 'cuda'

def model_is_on_cpu(model):
    '''Check if the model is on CPU.'''
    return model_device(model).type == 'cpu'

def move_model_to_gpu(model):
    '''Move the model to GPU if available.'''
    m =f'No GPU available. Model {model.base_model_prefix} will remain on CPU.'
    if model_is_on_gpu(model):
        print('Model is already on GPU.')
        return model
    if torch.cuda.is_available():
        print(f'Moving model {model.base_model_prefix} to GPU.')
        model.to('cuda')
    else: print('WARNING:',m)
    return model

def move_model_to_cpu(model):
    '''Move the model to CPU.'''
    if model_is_on_cpu(model):
        print('Model is already on CPU.')
        return model
    print(f'Moving model {model.base_model_prefix} to CPU.')
    model.to('cpu')
    return model

def move_model(model, gpu = False):
    '''Move the model to GPU if gpu is True, otherwise move it to CPU.'''
    if gpu: return move_model_to_gpu(model)
    else: return move_model_to_cpu(model)
    
def handle_model_feature_extractor(model, feature_extractor, gpu,
    do_move_model = False, cache_directory = None, 
    for_attention_extraction = False):
    '''Handle the loading of the model and feature extractor based on the inputs.
    model            A pretrained model or a string representing the model name.
    feature_extractor A feature extractor or None.
    gpu              If True, the model will be moved to GPU if available.
    do_move_model     If True, the model will be moved if gpu flag differs from
                     current device of the model. 
    cache_directory   Directory to cache the model if it needs to be loaded.
    '''
    if isinstance(model, str):
        model_name = model
    else:
        model_name = None
    if feature_extractor is None and model_name:
        feature_extractor = load_feature_extractor(model)
    if feature_extractor is None:
        if model_name is None and model is not None:
            model_name_or_path = getattr(model, 'name_or_path', None)
            if model_name_or_path and Path(model_name_or_path).exists():
                feature_extractor = load_feature_extractor(model_name_or_path)
            elif model_name_or_path and '/' in model_name_or_path:
                feature_extractor = load_feature_extractor(model_name_or_path)
            else:
                feature_extractor = load_feature_extractor()
        else:
            feature_extractor = load_feature_extractor()
    d = feature_extractor.to_dict()
    if d.get('feature_extractor_type') != 'Wav2Vec2FeatureExtractor':
        m = f'WARNING: Feature extractor {type(feature_extractor)} '
        m += 'may not be supported. A Wav2Vec2FeatureExtractor is expected.'
        print(m)
    if for_attention_extraction: loader = load_model_for_attention_extraction
    else: loader = load_pretrained_model
    if model is None: model = loader(gpu = gpu,cache_directory = cache_directory)
    elif model_name: model = loader(model_name, gpu = gpu,   
        cache_directory = cache_directory)
    if not model_registry.is_supported_model(model):
        smt = model_registry.SUPPORTED_MODEL_TYPES
        models = '\n'.join(['\t' + str(x) for x in smt])
        m = f'WARNING: Model {type(model)} may not be supported. '
        m += 'Make sure it is one of the supported models: \n'
        m += f'{models}'
        print(m)
    if for_attention_extraction:
        if model.config._attn_implementation != 'eager':
            m = f'WARNING: Model {model.base_model_prefix} is not configured '
            m += 'for eager attention extraction. Attention extraction may not '
            m += 'work as expected.'
            print(m)
    if do_move_model: move_model(model, gpu)
    gpu = model_is_on_gpu(model)
    return model, feature_extractor, gpu
