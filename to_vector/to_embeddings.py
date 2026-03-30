from . import audio
from . import load
from pathlib import Path
import torch
from transformers.modeling_outputs import BaseModelOutput


def audio_to_vector(audio_array, model = None, gpu = False,
    numpify_output = True):
    '''Convert an audio array to a vector using a pretrained model.
    audio_array            A 1D numpy array containing the audio samples.
    model                  A pretrained model. If None, the default model 
                           will be loaded. If a string is provided, it will be
                           used to load the model.
    gpu                    If True, the model and inputs will be moved to 
                           GPU if available.
    numpify_output         If True, the output will be converted to numpy arrays.
    '''
    model = load.prepare_model(model, gpu)
    model_type = load.get_model_type(model)
    if model_type == 'spidr':
        return _spidr_audio_to_vector(audio_array, model, numpify_output)
    return _huggingface_audio_to_vector(audio_array, model, model_type,
        numpify_output)


def _huggingface_audio_to_vector(audio_array, model, model_type,
    numpify_output=True):
    '''Convert an audio array with a Hugging Face feature extractor.'''
    feature_extractor = load.prepare_feature_extractor(model)
    gpu = load.model_is_on_gpu(model)
    inputs = feature_extractor(audio_array, sampling_rate=16_000,
        return_tensors='pt', padding=True)
    if gpu: inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    if hasattr(outputs, 'last_hidden_state'):
        outputs.last_hidden_state = None
    outputs.model_type = model_type
    if not hasattr(outputs, 'extract_features'):
        if model_type == 'hubert':
            o = audio_to_cnn(audio_array, model, gpu)
            outputs.extract_features = o
    if numpify_output: return numpify(outputs)
    return outputs


def _spidr_audio_to_vector(audio_array, model, numpify_output=True):
    '''Convert an audio array with SpidR-specific frontend logic.'''
    x = audio.standardize_audio(audio_array)
    x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
    if load.model_is_on_gpu(model): x = x.to('cuda')
    with torch.no_grad():
        extract_features = model.feature_extractor(x)
        extract_features = model.feature_projection(extract_features)
        hidden_states = model.student.get_intermediate_outputs(
            extract_features)
        codebook_predictions = model.get_codebooks(x)
    outputs = BaseModelOutput(
        hidden_states=tuple(hidden_states))
    outputs.extract_features = extract_features
    outputs.model_type = 'spidr'
    if numpify_output: return numpify(outputs)
    return outputs


def filename_to_vector(audio_filename, start=0.0, end=None,
    model=None, gpu = False, identifier = '', name = '',
    numpify_output = True):
    '''Convert an audio file to a vector using a pretrained model.
    audio_filename         Path to the audio file.
    start                  Start time in seconds. Default is 0.0.
    end                    End time in seconds. Default is None, which means
                           the end of the file.
    model                  A pretrained model. If None, the default model
    identifier             An optional identifier to add to the output.
    name                   An optional name to add to the output.
    numpify_output         If True, the output will be converted to numpy arrays.
    '''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    outputs = audio_to_vector(array, model, gpu, numpify_output)
    outputs = add_info(outputs, audio_filename, start, end, identifier, name)
    return outputs

def add_info(outputs, audio_filename, start, end, identifier, name):
    '''Add information about the audio file to the output object.
    outputs            The output object to which the information will be added.
    audio_filename     The path to the audio file.
    start              The start time of the audio segment.
    end                The end time of the audio segment.
    identifier         An optional identifier to add to the output.
    name               An optional name to add to the output.
    '''
    audio_filename = str(audio_filename)
    outputs.audio_filename = audio_filename
    outputs.start_time = start
    outputs.end_time = end
    outputs.identifier = identifier
    outputs.name = name
    return outputs

def numpify(outputs):
    '''Convert the outputs of a model to numpy arrays.
    outputs            The output object from the model.
    '''
    if hasattr(outputs, 'extract_features'):
        if type(outputs.extract_features) == torch.Tensor:
            outputs.extract_features = outputs.extract_features.cpu().numpy()
    hs = []
    for hidden_state in outputs.hidden_states:
        hs.append(hidden_state.cpu().numpy())
    outputs.hidden_states = hs
    return outputs


def audio_to_cnn(audio, model=None, gpu = False, identifier = '',
    name = ''):
    '''Convert an audio array to features using a pretrained model.
    model                A pretrained model. If None, the default model
                         if string is provided it will be used to load the model
    gpu                  If True, the model and inputs will be moved to 
                         GPU if available.
    identifier           An optional identifier to add to the output.
    name                 An optional name to add to the output.
    '''
    model = load.prepare_model(model, gpu)
    feature_extractor = load.prepare_feature_extractor(model)
    gpu = load.model_is_on_gpu(model)
    array = audio
    inputs = feature_extractor(array, sampling_rate=16_000, return_tensors='pt',
        padding= True)
    if gpu:inputs = inputs.to('cuda')
    with torch.no_grad():
        input_values = inputs['input_values']
        if 'ForPreTraining' in str(type(model)):
            outputs = model.wav2vec2.feature_extractor(input_values)
        else:
            outputs = model.feature_extractor(input_values)
    outputs = outputs.transpose(1,2).detach().cpu().numpy()
    return outputs

def filename_to_cnn(audio_filename, start=0.0, end=None,
    model=None, gpu = False, identifier = '', name = ''):
    '''Convert an audio file to features using a pretrained model.
    audio_filename         Path to the audio file.
    start                  Start time in seconds. Default is 0.0.
    end                    End time in seconds. Default is None, which means
                           the end of the file.
    model                  A pretrained model. If None, the default model
    gpu                    If True, the model and inputs will be moved to 
                           GPU if available.
    identifier             An optional identifier to add to the output.
    name                   An optional name to add to the output.
    '''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    outputs = audio_to_cnn(array, model, gpu, identifier, name)
    o = BaseModelOutput(hidden_states = None)
    o.extract_features = outputs
    outputs = add_info(o, audio_filename, start, end, identifier, name)
    return outputs
