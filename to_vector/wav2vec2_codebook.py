from pathlib import Path

import numpy as np
import torch

from . import audio
from . import batch_helper
from . import hf_batch_helper
from . import load
from . import to_embeddings


def filename_to_codebook_indices(audio_filename, start=0.0, end=None,
    model_pt=None, gpu=False):
    '''Convert an audio file to codebook indices using a pretrained model.
    audio_filename         Path to the audio file.
    start                  Start time in seconds. Default is 0.0.
    end                    End time in seconds. Default is None, which means
                            the end of the file.
    model_pt               A pretrained Wav2Vec2ForPreTraining model which has
                           the codebook. If None, the default model will be
                           used.
    gpu:                   whether to request CUDA
    '''
    array = audio.load_audio(audio_filename, start, end)
    codebook_indices = audio_to_codebook_indices(array, model_pt, gpu)
    return codebook_indices


def filename_to_codevectors(audio_filename, start=0.0, end=None,
    model_pt=None, gpu=False):
    '''Convert an audio file to codevectors using a pretrained model.
    audio_filename         Path to the audio file.
    start                  Start time in seconds. Default is 0.0.
    end                    End time in seconds. Default is None, which means
                            the end of the file.
    model_pt               A pretrained Wav2Vec2ForPreTraining model which has
                           the codebook. If None, the default model will be
                           used.
    gpu:                   whether to request CUDA
    '''
    array = audio.load_audio(audio_filename, start, end)
    codevectors = audio_to_codevectors(array, model_pt, gpu)
    return codevectors


def filename_batch_to_codebook_indices(audio_filenames, starts=None,
    ends=None, model_pt=None, gpu=False, batch_size=None):
    '''Convert multiple audio files to Wav2Vec2 codebook indices.'''
    outputs = list(iter_filename_batch_to_codebook_indices(audio_filenames,
        starts=starts, ends=ends, model_pt=model_pt, gpu=gpu,
        batch_size=batch_size))
    if len(audio_filenames) != len(outputs):
        m = f'iter_filename_batch_to_codebook_indices() returned '
        m += f'{len(outputs)} outputs, but expected {len(audio_filenames)}'
        raise ValueError(m)
    return outputs


def iter_filename_batch_to_codebook_indices(audio_filenames, starts=None,
    ends=None, model_pt=None, gpu=False, batch_size=None):
    '''Yield Wav2Vec2 codebook indices for audio files in input order.'''
    model_pt = _prepare_model_pt(model_pt, gpu)
    codebook = load_codebook(model_pt)
    for codevectors in iter_filename_batch_to_codevectors(audio_filenames,
        starts=starts, ends=ends, model_pt=model_pt, gpu=False,
        batch_size=batch_size):
        yield codevectors_to_codebook_indices(
            codevectors[np.newaxis, ...], codebook)


def filename_batch_to_codevectors(audio_filenames, starts=None, ends=None,
    model_pt=None, gpu=False, batch_size=None):
    '''Convert multiple audio files to Wav2Vec2 codevectors.'''
    outputs = list(iter_filename_batch_to_codevectors(audio_filenames,
        starts=starts, ends=ends, model_pt=model_pt, gpu=gpu,
        batch_size=batch_size))
    if len(audio_filenames) != len(outputs):
        m = f'iter_filename_batch_to_codevectors() returned {len(outputs)} '
        m += f'outputs, but expected {len(audio_filenames)}'
        raise ValueError(m)
    return outputs


def iter_filename_batch_to_codevectors(audio_filenames, starts=None,
    ends=None, model_pt=None, gpu=False, batch_size=None):
    '''Yield Wav2Vec2 codevectors for audio files in input order.'''
    audio_filenames = [Path(filename).resolve()
        for filename in audio_filenames]
    if not audio_filenames: return
    _check_batch_lengths(starts, ends, audio_filenames)
    starts = _check_batch_values(starts, len(audio_filenames), 0.0, 'starts')
    ends = _check_batch_values(ends, len(audio_filenames), None, 'ends')
    model_pt = _prepare_model_pt(model_pt, gpu)
    model_on_gpu = load.model_is_on_gpu(model_pt)
    input_items = zip(audio_filenames, starts, ends)
    for batch in batch_helper.split(input_items, batch_size=batch_size):
        arrays = [audio.load_audio(filename, start, end)
            for filename, start, end in batch]
        yield from _audio_batch_to_codevectors(arrays, model_pt)
        del arrays
        if model_on_gpu: torch.cuda.empty_cache()


def audio_to_codevectors(audio, model_pt=None, gpu=False):
    '''map an audio array to codevectors
    audio           is a numpy array of the audio signal
    model_pt        is the Wav2Vec2ForPreTraining model which has the codebook
    gpu             whether to use gpu or not
    '''
    if model_pt is None:
        model_pt = load.load_model_pt(gpu=gpu)
    cnn = to_embeddings.audio_to_cnn(audio, model_pt, gpu)
    return cnn_output_to_codevectors(cnn, model_pt)

def audio_to_codebook_indices(audio, model_pt=None, gpu=False):
    '''map an audio array to codebook indices
    audio           is a numpy array of the audio signal
    model_pt        is the Wav2Vec2ForPreTraining model which has the codebook
    gpu             whether to use gpu or not
    '''
    if model_pt is None:
        model_pt = load.load_model_pt(gpu=gpu)
    codevectors = audio_to_codevectors(audio, model_pt, gpu)
    codebook = load_codebook(model_pt)
    codebook_indices = codevectors_to_codebook_indices(codevectors, codebook)
    return codebook_indices


def _audio_batch_to_codevectors(audio_arrays, model_pt):
    '''Quantize one padded batch and return per-item codevectors.'''
    cnn_outputs, output_lengths = _audio_batch_to_cnn_outputs(
        audio_arrays, model_pt)
    with torch.no_grad():
        codevectors, tensor = model_pt.quantizer(cnn_outputs)
    codevectors = codevectors.detach().cpu().numpy()
    del cnn_outputs
    del tensor
    items = [
        codevectors[index, :output_length]
        for index, output_length in enumerate(output_lengths)
    ]
    del codevectors
    return items


def _audio_batch_to_cnn_outputs(audio_arrays, model_pt):
    '''Return padded CNN outputs and valid frame lengths for each item.'''
    feature_extractor = load.prepare_feature_extractor(model_pt)
    gpu = load.model_is_on_gpu(model_pt)
    arrays = [np.asarray(audio_array) for audio_array in audio_arrays]
    inputs = feature_extractor(arrays, sampling_rate=16_000,
        return_tensors='pt', padding=True)
    del arrays
    if gpu: inputs = inputs.to('cuda')
    with torch.no_grad():
        input_values = inputs['input_values']
        outputs = model_pt.wav2vec2.feature_extractor(input_values)
    outputs = outputs.transpose(1, 2).detach()
    output_lengths = _compute_cnn_output_lengths(inputs, outputs, model_pt)
    del inputs
    del input_values
    return outputs, output_lengths


def _compute_cnn_output_lengths(inputs, outputs, model_pt):
    '''Resolve valid Wav2Vec2 CNN output lengths for padded inputs.'''
    if 'attention_mask' not in inputs:
        return [int(outputs.shape[1])] * int(outputs.shape[0])
    input_lengths = inputs['attention_mask'].sum(dim=-1).to('cpu')
    if hasattr(model_pt, '_get_feat_extract_output_lengths'):
        lengths = model_pt._get_feat_extract_output_lengths(input_lengths)
    elif hasattr(model_pt, 'wav2vec2') and hasattr(
        model_pt.wav2vec2, '_get_feat_extract_output_lengths'):
        lengths = model_pt.wav2vec2._get_feat_extract_output_lengths(
            input_lengths)
    else:
        lengths = [hf_batch_helper.n_frames(int(length))
            for length in input_lengths]
    return [int(length) for length in lengths]


def _prepare_model_pt(model_pt, gpu):
    if model_pt is None:
        return load.load_model_pt(gpu=gpu)
    return model_pt


def _check_batch_values(values, expected_length, default, name):
    if values is None:
        return [default] * expected_length
    values = list(values)
    if len(values) != expected_length:
        m = f'{name} must have the same length as audio_filenames'
        raise ValueError(m)
    return values


def _as_model_tensor(value, model_pt):
    try:
        device = load.model_device(model_pt)
    except (AttributeError, StopIteration):
        return torch.as_tensor(value)
    return torch.as_tensor(value, device=device)

def outputs_to_codebook_indices(outputs, model_pt):
    '''map wav2vec2 outputs to codebook indices
    outputs     is the hidden states output of the wav2vec2 model
    model_pt    is the Wav2Vec2ForPreTraining model which has the codebook
    '''
    cv = outputs_to_codevectors(outputs, model_pt)
    if cv.ndim == 2: cv = cv[np.newaxis, ...]
    codebook = load_codebook(model_pt)
    ci = codevectors_to_codebook_indices(cv, codebook)
    return ci

def cnn_output_to_codevectors(cnn_output, model_pt, codebook=None):
    '''map cnn output to codebook indices
    cnn_output  is the output of the cnn (i.e. extract_features) 
                of the wav2vec2 model
    model_pt    is the Wav2Vec2ForPreTraining model which has the codebook
    codebook    is the codebook to use, if None it will be loaded from the model
    '''
    cnn_output = _as_model_tensor(cnn_output, model_pt)
    m = 'cnn output has more than one batch, please provide a single batch '
    m += 'of cnn output'
    if cnn_output.ndim == 1: cnn_output = cnn_output.view(1, 1, -1)
    elif cnn_output.ndim == 2:
        cnn_output = cnn_output.view(1, cnn_output.shape[0], -1)
    elif cnn_output.ndim == 3:
        pass
    else:
        raise ValueError(f'cnn output has {cnn_output.ndim} dimensions (<4)')
    codevectors, tensor = model_pt.quantizer(cnn_output)
    codevectors = codevectors.detach().cpu().numpy()
    del cnn_output
    del tensor
    return codevectors

def cnn_output_to_codebook_indices(cnn_output, model_pt, codebook=None):
    codevectors = cnn_output_to_codevectors(cnn_output, model_pt, codebook)
    if codebook is None:
        codebook = load_codebook(model_pt)
    ci = codevectors_to_codebook_indices(codevectors, codebook)
    return ci

def outputs_to_codevectors(outputs, model_pt):
    '''map cnn outputs to codevectors
    outputs     is the hidden states output of the wav2vec2 model
    model_pt    is the Wav2Vec2ForPreTraining model which has the codebook
                and quantizer loaded
    '''
    if type(outputs.extract_features) == np.ndarray:
        cnn_output = _as_model_tensor(outputs.extract_features, model_pt)
    else:
        cnn_output = outputs.extract_features
    codevectors, tensor = model_pt.quantizer(cnn_output)
    codevectors = codevectors.detach().cpu().numpy()[0]
    del cnn_output
    del tensor
    return codevectors

def load_codebook(model_pt):
    '''load the codebook from the model'''
    codebook = model_pt.quantizer.codevectors
    return codebook.detach().cpu().numpy()[0]

def codevectors_to_codebook_indices(codevectors, codebook):
    '''map codevectors to codebook indices
    codevectors     is a list of codevectors a codevector is a quantized
                    representation of the cnn output (i.e. extract_features)
    codebook        a matrix of codevectors, each quantized representation
                    can be found in the codebook, there are two codebooks
                    so a complete codevector can be represented with two indices
                    i.e. the locations in the codebooks
    '''
    batches = []
    for batch_index in range(codevectors.shape[0]):
        codebook_indices = []
        for codevector in codevectors[batch_index]:
            ci = codevector_to_codebook_indices(codevector, codebook)
            codebook_indices.append(ci)
        batches.append(codebook_indices)
    if len(batches) == 1: return batches[0]
    return batches

def codevector_to_codebook_indices(codevector, codebook):
    '''map a codevector to codebook indices
    codevector      a codevector is a quantized
                    representation of the cnn output (i.e. extract_features)
    codebook        a matrix of codevectors, each quantized representation
                    can be found in the codebook, there are two codebooks
                    so a complete codevector can be represented with two indices
                    i.e. the locations in the codebooks
    '''
    slice_index = codebook.shape[-1]
    q1, q2 = codevector[:slice_index], codevector[slice_index:]
    index1 = get_row_index_of_vector_in_matrix(q1, codebook)
    index2 = get_row_index_of_vector_in_matrix(q2, codebook)
    codebook_indices = (index1, index2)
    return codebook_indices

def multiple_codebook_indices_to_codevectors(codebook_indices, codebook):
    '''map multiple codebook indices to codevectors.
    codebook_indices   is a list of tuples of codebook indices, each tuple
                        contains the indices for the two codebooks
    codebook            a matrix of codevectors, each quantized representation
    '''
    if codebook is None:
        raise ValueError('please provide codebook')
    cv = []
    for ci in codebook_indices:
        cv.append(codebook_indices_to_codevector(ci, codebook))
    return np.array(cv)

def codebook_indices_to_codevector(codebook_indices, codebook):
    '''map codebook indices to a codevector
    codebook_indices   is a tuple of codebook indices, each tuple
                        contains the indices for the two codebooks
    codebook            a matrix of codevectors, each quantized representation
    '''
    if codebook is None:
        raise ValueError('please provide codebook')
    a = codebook[codebook_indices[0]]
    b = codebook[codebook_indices[1]]
    return np.hstack((a,b))


def get_row_index_of_vector_in_matrix(vector, matrix):
    '''find the row index of a vector in a matrix.
    vector  is the vector to find in the matrix
    matrix  is the matrix to search for the vector
    '''
    matches = np.argwhere(np.isclose(matrix, vector, rtol = 1e-5,
        atol = 1e-8).all(1)).flatten()
    if matches.size == 0:
        raise ValueError('vector was not found in the codebook')
    return matches[0]

def _check_batch_lengths(starts, ends, audio_filenames):
    if starts is None and ends is None: pass
    elif ends is None and len(audio_filenames) == len(starts): pass
    elif starts is None and len(audio_filenames) == len(ends): pass
    elif len(audio_filenames) == len(starts) == len(ends): pass
    else:
        raise ValueError('filenames, starts, and ends must have the same length')
