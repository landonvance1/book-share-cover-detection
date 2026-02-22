# 002 — OpenVINO Acceleration Abandoned

**Date:** 2026-02-22
**Status:** Rejected

---

## Context

Timing data from the ONNX pipeline showed the vision encoder consuming 40–65% of total
inference time (~1.2–3.6 s per image on a 6-thread CPU). On Intel hardware,
`onnxruntime-openvino` exposes an `OpenVINOExecutionProvider` (OV EP) that offloads ONNX
graph execution to Intel's OpenVINO runtime, applying Intel-specific graph optimisations
and optionally targeting iGPU or NPU. The vision encoder (a ViT-style DaViT architecture)
was the primary candidate for acceleration.

`onnxruntime` and `onnxruntime-openvino` cannot coexist — the swap requires a full
package replacement.

---

## What Was Attempted

- Provider-selection logic to route each `InferenceSession` through OV EP with CPU
  fallback
- Pre-flight model inspection (`_resolve_providers`) to skip OV EP for sessions containing
  `MatMulNBits` (INT4 op unsupported by OV EP 1.23.0)
- `cache_dir` injection into OV EP options — in standalone scripts this converted hard
  C++ crashes into catchable Python exceptions
- Subprocess probe (`_probe_openvino`) — loads each model in a child process so a segfault
  kills the child, not the main process, enabling graceful CPU fallback

---

## Problems

### 1. MatMulNBits (INT4) not supported — OV EP 1.23.0

Q4-quantised models use the `MatMulNBits` operator (ONNX opset 21). OV EP 1.23.0 does
not implement this op and crashes on session creation. With Q4 models, all four sessions
fell back to CPU — no OpenVINO benefit at all.

### 2. DaViT vision encoder segfaults OV EP's graph compiler

Switching to float32 models (no `MatMulNBits`) produced a hard segfault inside OV EP's
C++ graph compiler when loading `vision_encoder.onnx`. The DaViT window-attention ops
(`Range`, `Mod`, `Tile`, `Pad`) trigger a corrupted node-index in OV EP 1.23.0:

```
node_index < nodes_.size() was false. Got: 10546600751561601625 Max: 2864
```

The `cache_dir` trick converted this to a Python exception in standalone scripts but not
reliably inside the app process. The subprocess probe correctly detected the crash and fell
back to CPU — but that means the bottleneck session (vision encoder, 40–65% of time)
cannot use OpenVINO at all, eliminating the primary motivation.

### 3. PyTorch uninstall broke the processor

PyTorch was uninstalled during debugging on the assumption it was causing library
conflicts. Florence-2's custom processor (`processing_florence2.py`) uses `torch.Tensor`
as a runtime type annotation at class-definition time. Without PyTorch, the processor
imported but produced incorrect outputs: the decoder generated tokens up to the 1024-token
limit on every image and post-processing returned empty text. PyTorch must remain
installed.

---

## Decision

**Do not pursue OpenVINO acceleration at this time.** Revert all OV EP code.

Even if the DaViT segfault were fixed in a future OV EP release, the vision encoder is the
bottleneck. Without it running under OpenVINO, the remaining sessions (embed_tokens,
encoder, decoder) represent only 35–60% of total time, with the decode loop further split
across ~1024 individual session calls — a poor fit for OV EP's per-graph optimisation.

---

## Revisit Conditions

- `onnxruntime-openvino` releases a fix for the DaViT node-index bug (monitor
  [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases))
- The vision encoder is re-exported without DaViT window-attention ops
- Direct OpenVINO Model Optimizer conversion (bypassing ORT's OV EP entirely) is explored
- NPU deployment is investigated — Florence-2 base (0.23B params) may fit on an Intel NPU

The full implementation is preserved in git history on branch
`12-export-florence-2-to-onnx` for reference.
