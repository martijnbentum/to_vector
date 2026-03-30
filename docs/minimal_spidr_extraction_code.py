import torch
import soundfile as sf

def load_and_standardize(path):
    '''load mono wav and standardize.
    path                    wav file path
    '''
    x, sr = sf.read(path)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = torch.tensor(x, dtype=torch.float32)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x

def forward_features(model, waveform, device='cpu'):
    '''run model and return student, teacher and codebook info.
    model                   spidr model
    waveform                1d tensor
    device                  cpu or cuda
    '''
    model = model.to(device).eval()
    x = waveform.to(device).unsqueeze(0)

    with torch.inference_mode():
        # shared frontend
        feats = model.feature_extractor(x)
        feats = model.feature_projection(feats)

        # student hidden states (ALL layers)
        student_states = model.student.get_intermediate_outputs(feats)

        # teacher hidden states (ALL layers)
        teacher_states = model.teacher.get_intermediate_outputs(feats)

        # codebook predictions (student side)
        codebook_preds = model.get_codebooks(x)

    return student_states, teacher_states, codebook_preds

def get_codebooks(model):
    '''return actual stored codebook vectors.
    model                   spidr model
    '''
    return [cb.codebook for cb in model.codebooks]

#example usage:
waveform = load_and_standardize('example.wav')

student, teacher, codebook_preds = forward_features(model, waveform, device='cpu')
codebooks = get_codebooks(model)

print('n student layers:', len(student))
print('n teacher layers:', len(teacher))
print('n codebooks:', len(codebooks))

print('student layer 6 shape:', student[5].shape)
print('teacher layer 12 shape:', teacher[11].shape)

print('codebook 0 shape:', codebooks[0].shape)
print('codebook prediction 0 shape:', codebook_preds[-1].shape)  # last codebook prediction
