from . import load
from . import to_embeddings
import numpy as np
import torch

def filename_to_codebook_indices(audio_filename, start=0.0, end=None,
    model_pt=None, feature_extractor = None, gpu = False):
    '''Convert an audio file to codebook indices using a pretrained model.
    audio_filename         Path to the audio file.
    start                  Start time in seconds. Default is 0.0.
    end                    End time in seconds. Default is None, which means
                            the end of the file.
    model_pt               A pretrained Wav2Vec2ForPreTraining model which has
                            the codebook. If None, the default model will be
                            used.
    feature_extractor      A feature extractor. If None, the default feature
                            extractor will be used.
    gpu                     If True, the model will be moved to GPU if available.
    '''
    array = load.load_audio(audio_filename,start, end)
    codebook_indices = audio_to_codebook_indices(array, model_pt, 
        feature_extractor, gpu)
    return codebook_indices

def filename_to_codevectors(audio_filename, start=0.0, end=None,
    model_pt=None, feature_extractor = None, gpu = False):
    '''Convert an audio file to codevectors using a pretrained model.
    audio_filename         Path to the audio file.
    start                  Start time in seconds. Default is 0.0.
    end                    End time in seconds. Default is None, which means
                            the end of the file.
    model_pt               A pretrained Wav2Vec2ForPreTraining model which has
                            the codebook. If None, the default model will be
                            used.
    feature_extractor      A feature extractor. If None, the default feature
                            extractor will be used.
    gpu                     If True, the model will be moved to GPU if available.
    '''
    array = load.load_audio(audio_filename,start, end)
    codevectors = audio_to_codevectors(array, model_pt, feature_extractor, gpu)
    return codevectors

def audio_to_codevectors(audio, model_pt = None, feature_extractor = None, 
    gpu = False):
    '''map an audio array to codevectors
    audio           is a numpy array of the audio signal
    model_pt        is the Wav2Vec2ForPreTraining model which has the codebook
    feature_extractor is the feature extractor to use
    gpu             whether to use gpu or not
    '''
    if model_pt is None:
        model_pt = load.load_model_pt(gpu=gpu)
    cnn = to_embeddings.audio_to_cnn(audio, model_pt, feature_extractor, gpu)
    return cnn_output_to_codevectors(cnn, model_pt)

def audio_to_codebook_indices(audio, model_pt = None, feature_extractor = None,
    gpu = False):
    '''map an audio array to codebook indices
    audio           is a numpy array of the audio signal
    model_pt        is the Wav2Vec2ForPreTraining model which has the codebook
    feature_extractor is the feature extractor to use
    gpu             whether to use gpu or not
    '''
    if model_pt is None:
        model_pt = load.load_model_pt(gpu=gpu)
    codevectors = audio_to_codevectors(audio, model_pt, feature_extractor, gpu)
    codebook = load_codebook(model_pt)
    codebook_indices = codevectors_to_codebook_indices(codevectors, codebook)
    return codebook_indices

def outputs_to_codebook_indices(outputs, model_pt):
    '''map wav2vec2 outputs to codebook indices
    outputs     is the hidden states output of the wav2vec2 model
    model_pt    is the Wav2Vec2ForPreTraining model which has the codebook
    '''
    cv = outputs_to_codevectors(outputs, model_pt)
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
    cnn_output = torch.from_numpy(cnn_output)
    m = 'cnn output has more than one batch, please provide a single batch of cnn output'
    if cnn_output.ndim == 1: cnn_output = cnn_output.view(1,1,-1)
    elif cnn_output.ndim == 2: 
        cnn_output = cnn_output.view(1,cnn_output.shape[0],-1)
    elif cnn_output.ndim == 3: pass
    else: 
        raise ValueError(f'cnn output has {cnn_output.ndim} dimensions (<4)')
    codevectors, tensor = model_pt.quantizer(cnn_output)
    codevectors = codevectors.detach().numpy()
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
        cnn_output = torch.from_numpy(outputs.extract_features)
    else:
        cnn_output = outputs.extract_features
    codevectors, tensor = model_pt.quantizer(cnn_output)
    return codevectors.detach().numpy()[0]

def load_codebook(model_pt):
    '''load the codebook from the model'''
    codebook = model_pt.quantizer.codevectors
    return codebook.detach().numpy()[0]

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
