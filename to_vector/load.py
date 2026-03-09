from decouple import config
from huggingface_hub import login
import librosa
import os
import torch
from transformers import AutoFeatureExtractor
from transformers import AutoProcessor
from transformers import AutoModel
from transformers import AutoModelForPreTraining

token = config('HF_TOKEN', default= None)
default_cache_directory = config('HF_HOME', default= '~/.cache/huggingface')
if default_cache_directory.startswith('~'):
    default_cache_directory = os.path.expanduser(default_cache_directory)
if token: login(token)
else: print('No Hugging Face token found. Set it in .env file under HF_TOKEN')

wav2vec2_base= 'facebook/wav2vec2-base'
hubert_base = 'facebook/hubert-base-ls960'
wavlm_base = 'microsoft/wavlm-base-plus'
default_checkpoint = wav2vec2_base

def load_audio(filename, start = 0.0, end=None):
	if not end: duration = None
	else: duration = end - start
	audio, sr = librosa.load(filename, sr = 16000, offset=start, 
        duration=duration)
	return audio

def load_processor(model_name_or_path):
    '''Load a processor. 
    model_name_or_path      Hugging Face repo id or local path
    '''
    return AutoProcessor.from_pretrained(model_name_or_path)

def load_feature_extractor(model_name_or_path):
    '''Load feature extractor.
    model_name_or_path      Hugging Face repo id or local path
    '''
    return AutoFeatureExtractor.from_pretrained(model_name_or_path) 

def load_pretrained_model(model_name_or_path = None, cache_directory = None, 
        gpu = False):
    if not model_name_or_path: model_name_or_path = default_checkpoint
    if not cache_directory: cache_directory = default_cache_directory
    model = AutoModel.from_pretrained(model_name_or_path,
        cache_dir = cache_directory)
    if gpu: model.to('cuda')
    return model

def load_model_pt(checkpoint = None, gpu = False):
    if not checkpoint: checkpoint = default_checkpoint
    model_pt = pt.from_pretrained(checkpoint)
    if gpu: model_pt.to('cuda')
    return model_pt

def load_hubert_base_model(cache_directory = None, gpu = False):
    return load_pretrained_model(hubert_base, cache_directory, gpu)

def load_wav2vec2_base_model(cache_directory = None, gpu = False):
    return load_pretrained_model(wav2vec2_base, cache_directory, gpu)

