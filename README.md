# to-vector

`to-vector` extracts representations from speech models in the Wav2Vec2 family.
It supports:

- hidden states
- convolutional feature encoder outputs
- attention tensors
- Wav2Vec2 codevectors and codebook indices
- SpidR codebook probabilities, indices, and codevectors

The package is designed for local scripts and research workflows where you want a
small wrapper around Hugging Face speech checkpoints without rebuilding the
preprocessing and tensor plumbing each time.

## Installation

From GitHub:

```bash
pip install git+https://git@github.com/martijnbentum/to-vector.git#egg=to-vector
```

With `uv`:

```bash
uv pip install git+https://git@github.com/martijnbentum/to-vector.git
```

For development:

```bash
pip install -e .[dev]
```

This keeps editable-install metadata in `.build/` instead of creating
`*.egg-info` at the repository root.

Create local Wav2Vec2-base and HuBERT-base checkpoints with their official
configurations and feature extractors, but randomly initialized weights:

```python
from to_vector.scripts import save_random_hubert_base
from to_vector.scripts import save_random_wav2vec2_base

save_random_wav2vec2_base('./wav2vec2-base-random')
save_random_hubert_base('./hubert-base-random')
```

## Supported model families

The library is built around Hugging Face models compatible with:

- `Wav2Vec2Model`
- `HubertModel`
- `WavLMModel`
- `Wav2Vec2ForPreTraining` for codebook-related helpers
- local `SpidR` checkpoints for SpidR-specific helpers

## Quick start

```python
import to_vector

outputs = to_vector.filename_to_vector("example.wav")
last_hidden_state = outputs.hidden_states[-1]
cnn_features = outputs.extract_features
```

Load audio once and reuse a model:

```python
import to_vector

audio = to_vector.load_audio("example.wav", start=0.5, end=1.5)
model = to_vector.load_model("facebook/wav2vec2-base")

outputs = to_vector.audio_to_vector(
    audio,
    model=model,
)
```

Extract convolutional features only:

```python
cnn = to_vector.filename_to_cnn("example.wav").extract_features
raw_cnn = to_vector.filename_to_cnn(
    "example.wav", normalize=False
).extract_features
```

CNN features are layer-normalized by default, matching `extract_features` from
full Wav2Vec2 and WavLM inference. Pass `normalize=False` to return the raw
convolutional output instead.

Extract attention tensors:

```python
attention = to_vector.filename_to_attention(
    "example.wav",
    model="facebook/wav2vec2-base",
    layer=0,
    average_heads=True,
).attentions
```

Extract vectors from a local SpidR checkpoint:

```python
import to_vector

outputs = to_vector.filename_to_vector(
    "example.wav",
    model="path/to/spidr-checkpoint.pt",
)
hidden_states = outputs.hidden_states
```

Extract student attention from a local SpidR checkpoint:

```python
import to_vector

attention = to_vector.filename_to_attention(
    "example.wav",
    model="path/to/spidr-checkpoint.pt",
    layer=0,
    average_heads=True,
).attentions
```

Extract codebook indices from a pretraining checkpoint:

```python
indices = to_vector.filename_to_codebook_indices("example.wav")
```

Extract SpidR codebook indices from a local checkpoint:

```python
from to_vector import spidr_codebook

indices = spidr_codebook.filename_to_codebook_indices(
    "example.wav",
    model="path/to/spidr-checkpoint.pt",
)
```

Extract batched SpidR codebook indices from a local checkpoint:

```python
from to_vector import spidr_codebook

items = spidr_codebook.filename_batch_to_codebook_indices(
    ["example-a.wav", "example-b.wav"],
    model="path/to/spidr-checkpoint.pt",
)
```

Extract SpidR codebook probabilities from a local checkpoint:

```python
from to_vector import spidr_codebook

probabilities = spidr_codebook.filename_to_codebook_probabilities(
    "example.wav",
    model="path/to/spidr-checkpoint.pt",
)
```

Extract batched SpidR codevectors for timed segments:

```python
from to_vector import spidr_codebook

codevectors = spidr_codebook.filename_batch_to_codevectors(
    ["example-a.wav", "example-b.wav"],
    starts=[0.0, 1.5],
    ends=[0.8, 2.1],
    model="path/to/spidr-checkpoint.pt",
)
```

## Output shapes

Exact dimensions depend on the checkpoint and input duration, but the API
returns these structures:

- `audio_to_vector` / `filename_to_vector`
  returns a Transformers output object with `hidden_states`, and usually
  `extract_features`
- `audio_to_cnn` / `filename_to_cnn`
  returns CNN features shaped like `(batch, frames, channels)`
- `audio_to_attention` / `filename_to_attention`
  returns an output object with `attentions`
- codebook helpers
  return numpy arrays or Python index tuples derived from the quantizer
- `spidr_codebook` helpers
  return numpy arrays shaped like `(frames, codebooks)` for indices and
  `(frames, codebooks, codebook_size)` for probabilities

By default, the convenience functions convert tensor outputs to numpy arrays.

## GPU behavior

Passing `gpu=True` requests CUDA, but the library now falls back cleanly to CPU
when CUDA is not available.

A CUDA context stays allocated until its process exits, so GPU memory cannot
be fully freed from a long-lived session such as IPython. The embedding
functions can therefore run in a child process that loads the model itself
and exits when done, which frees the GPU completely:

- `filename_batch_to_vector` and `iter_filename_batch_to_vector` do this
  automatically when `gpu=True` and `model` is a name, a path or `None`.
  Results stream back as each batch finishes.
- `filename_to_vector` and `audio_to_vector` do this only with `spawn=True`,
  because a loop over them would reload the model on every call.
- A loaded model object always runs in the current process.
- `spawn=False` disables the child process; `spawn=True` forces it.

```python
# runs in a child process; the GPU is free again when the loop ends
for outputs in to_vector.iter_filename_batch_to_vector(filenames,
    starts=starts, ends=ends, model='facebook/wav2vec2-base', gpu=True):
    ...
```

Scripts that start a child process must guard their entry point with
`if __name__ == '__main__':`, because the child re-imports the main module.

## Notes

- Audio is loaded with `librosa` at `16_000` Hz.
- `start` and `end` are interpreted in seconds.
- `end` must be greater than or equal to `start`.
- When you provide a loaded model object, the library tries to infer the correct
  feature extractor from `model.name_or_path`.

## SpidR Behavior

- SpidR-specific helpers standardize audio to zero mean and unit variance
  before converting it to tensors.
- `audio_to_vector` / `filename_to_vector` on SpidR return student hidden
  states and projected frontend features.
- `audio_to_attention` / `filename_to_attention` on SpidR return student
  attention only.
- The current local SpidR attention extraction targets the default SpidR base
  configuration, which uses 12 transformer layers and 12 attention heads
  according to the installed `SpidRConfig`.
- `spidr_codebook` reads the per-codebook probability outputs from SpidR and
  derives indices with `argmax`, returning one codebook column per selected
  SpidR layer.
- `audio_to_cnn` / `filename_to_cnn` are not implemented for SpidR yet.

## Backend Differences

- Hugging Face model families use a feature extractor loaded from the checkpoint
  metadata; SpidR helpers call the model frontend directly.
- Wav2Vec2 codebook helpers use quantizer codevectors from
  `Wav2Vec2ForPreTraining`; SpidR codebook helpers live in the
  `to_vector.spidr_codebook` module and return dense frame-by-codebook arrays.
- Hugging Face attention comes from the model's `output_attentions=True` path;
  SpidR attention is computed locally from the student transformer's attention
  projections because the upstream package does not expose attention weights.

## Public API

The package exposes its main helpers from the top-level module:

```python
import to_vector

to_vector.audio_to_vector
to_vector.filename_to_vector
to_vector.audio_to_attention
to_vector.filename_to_codebook_indices
```

## Testing

The repository includes a standard-library `unittest` suite, so tests can run
without installing extra tooling:

```bash
python3 -m unittest discover -s tests -v
```
