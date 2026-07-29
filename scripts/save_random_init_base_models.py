#!/usr/bin/env python3
'''Create and save randomly initialized wav2vec2-base and hubert-base
models. Architecture/config is copied from the official checkpoints but
weights are freshly initialized, not loaded from pretrained.
'''

from transformers import AutoFeatureExtractor
from transformers import HubertConfig
from transformers import HubertModel
from transformers import Wav2Vec2Config
from transformers import Wav2Vec2Model

wav2vec2_base = 'facebook/wav2vec2-base'
hubert_base = 'facebook/hubert-base-ls960'


def save_random_wav2vec2_base(
    output_dir='/vol/mlusers/mbentum/random/wav2vec2-base-random'):
    '''Create and save a randomly initialized wav2vec2-base model.'''
    config = Wav2Vec2Config.from_pretrained(wav2vec2_base)
    model = Wav2Vec2Model(config)
    model.save_pretrained(output_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(wav2vec2_base)
    feature_extractor.save_pretrained(output_dir)
    return model


def save_random_hubert_base(
    output_dir='/vol/mlusers/mbentum/random/hubert-base-random'):
    '''Create and save a randomly initialized hubert-base model.'''
    config = HubertConfig.from_pretrained(hubert_base)
    model = HubertModel(config)
    model.save_pretrained(output_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(hubert_base)
    feature_extractor.save_pretrained(output_dir)
    return model


if __name__ == '__main__':
    save_random_wav2vec2_base()
    save_random_hubert_base()
