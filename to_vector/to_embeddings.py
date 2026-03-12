from . import load
from pathlib import Path
import torch
from transformers.modeling_outputs import BaseModelOutput

def audio_to_vector(audio_array, model = None, feature_extractor = None,
    gpu = False, numpify_output = True):
    '''Convert an audio array to a vector using a pretrained model.
    audio_array            A 1D numpy array containing the audio samples.
    model                  A pretrained model. If None, the default model 
                           will be loaded. If a string is provided, it will be
                           used to load the model and feature extractor.
    feature_extractor      A feature extractor. If None, the default feature
                           extractor will be loaded based on the model. 
                           If a model name is provided,
                           it will be used to load the feature extractor. 
    gpu                    If True, the model and inputs will be moved to 
                           GPU if available.
    numpify_output         If True, the output will be converted to numpy arrays.
    '''
    model, feature_extractor, gpu = load.handle_model_feature_extractor(model, 
        feature_extractor, gpu)
    inputs = feature_extractor(audio_array, sampling_rate=16_000, 
        return_tensors='pt', padding= True)
    if gpu:inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs,output_hidden_states = True)
    if not hasattr(outputs, 'extract_features'):
        if 'Hubert' in str(type(model)):
            o = audio_to_cnn(audio_array, model, feature_extractor, gpu)
            outputs.extract_features = o
    if numpify_output:
        return numpify(outputs)
    return outputs

def filename_to_vector(audio_filename, start=0.0, end=None,
    model=None, feature_extractor = None, gpu = False,
        identifier = '', name = '', numpify_output = True):
    '''Convert an audio file to a vector using a pretrained model.
    audio_filename         Path to the audio file.
    start                  Start time in seconds. Default is 0.0.
    end                    End time in seconds. Default is None, which means
                           the end of the file.
    model                  A pretrained model. If None, the default model
    feature_extractor      A feature extractor. If None, the default feature
    identifier             An optional identifier to add to the output.
    name                   An optional name to add to the output.
    numpify_output         If True, the output will be converted to numpy arrays.
    '''
    audio_filename = Path(audio_filename).resolve()
    array = load.load_audio(audio_filename,start, end)
    outputs = audio_to_vector(array, model, feature_extractor, gpu,
        numpify_output)
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

def audio_to_cnn(audio, model=None, feature_extractor = None,
    gpu = False, identifier = '', name = ''):
    '''Convert an audio array to features using a pretrained model.
    model                A pretrained model. If None, the default model
                         if string is provided it will be used to load the model 
                         and feature extractor.    
    feature_extractor    A feature extractor. If None, the default feature
    gpu                  If True, the model and inputs will be moved to 
                         GPU if available.
    identifier           An optional identifier to add to the output.
    name                 An optional name to add to the output.
    '''
    model, feature_extractor, gpu = load.handle_model_feature_extractor(model, 
        feature_extractor, gpu)
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
    model=None, feature_extractor = None, gpu = False,
        identifier = '', name = ''):
    '''Convert an audio file to features using a pretrained model.
    audio_filename         Path to the audio file.
    start                  Start time in seconds. Default is 0.0.
    end                    End time in seconds. Default is None, which means
                           the end of the file.
    model                  A pretrained model. If None, the default model
    feature_extractor      A feature extractor. If None, the default feature
    gpu                    If True, the model and inputs will be moved to 
                           GPU if available.
    identifier             An optional identifier to add to the output.
    name                   An optional name to add to the output.
    '''
    audio_filename = Path(audio_filename).resolve()
    array = load.load_audio(audio_filename,start, end)
    outputs = audio_to_cnn(array, model, feature_extractor, gpu,
        identifier, name)
    o = BaseModelOutput(hidden_states = None)
    o.extract_features = outputs
    outputs = add_info(o, audio_filename, start, end, identifier, name)
    return outputs




