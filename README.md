# to-vector

`to-vector` extracts representations from speech models in the Wav2Vec2 family.
It supports:

- hidden states
- convolutional feature encoder outputs
- attention tensors
- Wav2Vec2 codevectors and codebook indices

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

## Supported model families

The library is built around Hugging Face models compatible with:

- `Wav2Vec2Model`
- `HubertModel`
- `WavLMModel`
- `Wav2Vec2ForPreTraining` for codebook-related helpers

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
```

Extract attention tensors:

```python
attention = to_vector.filename_to_attention(
    "example.wav",
    model="facebook/wav2vec2-base",
    layer=0,
    average_heads=True,
).attentions
```

Extract codebook indices from a pretraining checkpoint:

```python
indices = to_vector.filename_to_codebook_indices("example.wav")
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

By default, the convenience functions convert tensor outputs to numpy arrays.

## GPU behavior

Passing `gpu=True` requests CUDA, but the library now falls back cleanly to CPU
when CUDA is not available.

## Notes

- Audio is loaded with `librosa` at `16_000` Hz.
- `start` and `end` are interpreted in seconds.
- `end` must be greater than or equal to `start`.
- When you provide a loaded model object, the library tries to infer the correct
  feature extractor from `model.name_or_path`.

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
