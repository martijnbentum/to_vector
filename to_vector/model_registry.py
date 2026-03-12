from transformers import HubertModel
from transformers import Wav2Vec2Model
from transformers import WavLMModel

SUPPORTED_MODEL_TYPES = (Wav2Vec2Model, WavLMModel, HubertModel)


def is_supported_model(model):
    return isinstance(model, SUPPORTED_MODEL_TYPES)
