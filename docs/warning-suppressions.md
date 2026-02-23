# Test Warning Suppressions

Warnings suppressed in `pyproject.toml` under `[tool.pytest.ini_options] filterwarnings`.
Each entry documents why it was suppressed and what would allow it to be removed.

---

## 1. `DeprecationWarning` from `importlib._bootstrap` (SWIG/onnxruntime)

```
<frozen importlib._bootstrap>:488: DeprecationWarning:
builtin type SwigPyPacked has no __module__ attribute
```

**Source:** `onnxruntime`'s SWIG-generated C extension types do not set `__module__`.
Python 3.12 added a check for this and emits a `DeprecationWarning`; a future Python
version will make it a hard error.

**Why suppressed:** Not actionable — the types are in compiled C extensions inside
`onnxruntime`. We cannot add `__module__` to them.

**What removes it:** `onnxruntime` fixing their SWIG bindings in a future release.
Track: https://github.com/microsoft/onnxruntime/issues

---

## 2. `SyntaxWarning` for invalid escape sequence `\d` (GLiNER)

```
<unknown>:499: SyntaxWarning: invalid escape sequence '\d'
```

**Source:** GLiNER source code contains an unescaped `\d` in a regular string literal
(should be `r"\d"` or `"\\d"`). Python 3.12 deprecated unrecognised escape sequences;
a future Python version will raise `SyntaxError`, which would prevent `import gliner`
entirely.

**Why suppressed:** Bug in GLiNER's source — not actionable from our code.

**What removes it:** GLiNER fixing the escape sequence in a future release.
Track: https://github.com/urchade/GLiNER/issues

---

## 3. `DeprecationWarning` for `torch.jit.script` (PyTorch via GLiNER)

```
torch/jit/_script.py:1480: DeprecationWarning:
`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
```

**Source:** GLiNER uses `torch.jit.script` internally to script model components
during model load. PyTorch is deprecating this API in favour of `torch.compile` /
`torch.export`.

**Why suppressed:** GLiNER's internal implementation detail — not actionable from
our code. No current functional impact; `torch.jit.script` still works.

**What removes it:** GLiNER migrating to `torch.compile` in a future release.
Track: https://github.com/urchade/GLiNER/issues

---

## 4. `FutureWarning` for `resume_download` (huggingface_hub via GLiNER)

```
huggingface_hub/file_download.py:949: FutureWarning:
`resume_download` is deprecated and will be removed in version 1.0.0.
```

**Source:** GLiNER or transformers calls `hf_hub_download(..., resume_download=True)`,
a parameter being removed in `huggingface_hub` 1.0.0. Only fires on first run when
the model is not yet cached.

**Why suppressed:** Calling code is inside GLiNER/transformers — not actionable from
our code. The `transformers<5.0.0` pin in `requirements.txt` also constrains
`huggingface_hub` to `<1.0.0`, so removal of the parameter is not imminent.

**What removes it:** GLiNER/transformers stopping use of `resume_download`. Naturally
resolved if/when the `transformers<5.0.0` pin is lifted and GLiNER has fixed its
`transformers` 5.x compatibility (see `docs/decisions/002-nlp-engine-selection.md`).

---

## 5. `UserWarning` for sentencepiece byte fallback (transformers)

```
transformers/convert_slow_tokenizer.py:566: UserWarning:
The sentencepiece tokenizer that you are converting to a fast tokenizer uses the
byte fallback option which is not implemented in the fast tokenizers.
```

**Source:** GLiNER's tokenizer uses sentencepiece with byte fallback enabled. When
converting to the fast (Rust-based) tokenizer, `transformers` warns that unknown
characters will produce `[UNK]` rather than a sequence of byte tokens.

**Why suppressed:** Minor functional difference only affects non-ASCII input. Book
cover text is overwhelmingly ASCII; this has no observable impact on extraction
quality for our use case.

**What removes it:** Either (a) GLiNER switching to a tokenizer without byte fallback,
or (b) the `tokenizers` Rust library implementing byte fallback support. No near-term
fix expected.
