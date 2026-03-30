# to-vector status and recommendations (30-03-2026)

## Current status

- The repository is clean and the current `unittest` suite passes.
- The package presents a small, focused public API through
  `to_vector/__init__.py`.
- Core functionality is in place for hidden states, CNN features,
  attentions, and codebook-related outputs.
- The README is reasonably aligned with the package purpose and documents
  the main helpers.
- The project is still lightweight, with a small codebase and limited
  abstraction overhead.

## Possible loose ends

- Test coverage is narrow. The suite mostly checks exports and a few helper
  behaviors, but it does not exercise the main embedding, attention, and
  codebook flows with mocks end to end.
- `load_processor()` exists in `to_vector/load.py` but is not part of the
  public API and is not mentioned in the README. It may be dead weight unless
  it is intended for future use.
- Attention extraction is documented as a public capability, but the tests do
  not cover `audio_to_attention()`, `filename_to_attention()`,
  `outputs_to_attention()`, or the layer/head selection behavior.
- `filename_to_attention()` does not attach file metadata with `add_info()`,
  while `filename_to_vector()` and `filename_to_cnn()` do. That is a public
  API inconsistency.
- Codebook helpers accept `model_pt`, while most other helpers use `model`.
  The distinction makes sense internally, but it adds API friction and should
  either be documented more explicitly or wrapped more consistently.
- `model_registry.py` only treats base model classes as supported. Some code
  paths also handle pretraining models, so the support story is broader than
  the registry suggests.
- Several functions print warnings directly. That works for scripts, but it is
  harder to control than standard warnings or logging in larger workflows.
- Style is somewhat inconsistent in modified files: spacing, one-line
  conditionals, and compact inline statements reduce readability.
- `pyproject.toml` still describes the package as "A tool to extract hidden
  states from models.", which undersells the broader current scope.

## Recommendations

1. Expand tests around the real public API.
   Add mock-based tests for `audio_to_vector`, `filename_to_vector`,
   `audio_to_attention`, `filename_to_attention`, and the codebook helpers.
   The main gap is behavioral coverage, not raw line count.

2. Make filename-based helpers consistent.
   `filename_to_attention()` should probably mirror the metadata behavior of
   `filename_to_vector()` and `filename_to_cnn()` unless there is a clear
   reason not to.

3. Tighten the model support contract.
   Align `model_registry.py`, the README, and runtime checks so they describe
   the same supported model families, including any pretraining-only paths.

4. Replace `print()` warnings with `warnings.warn()` or logging.
   That keeps script usability while giving callers more control.

5. Clarify the public API in docs.
   Document which helpers return Transformers output objects versus numpy
   arrays versus Python tuples, and call out the `model` versus `model_pt`
   distinction more directly.

6. Clean up minor implementation rough edges.
   Normalize formatting in touched files, remove or justify unused helpers like
   `load_processor()`, and improve a few docstrings where current behavior is
   broader than the text suggests.

7. Improve package metadata.
   Update the package description in `pyproject.toml` to reflect embeddings,
   attentions, CNN features, and codebook utilities rather than only hidden
   states.

## Suggested order

1. Broaden test coverage.
2. Fix the attention metadata inconsistency.
3. Align support checks and documentation.
4. Clean up warnings and package metadata.
