# BootLegLM — Evidence Status

Checked in this workspace on 2026-11-07.

## Toolchains

- .NET SDK 8.0.427
- Microsoft.NETCore.App 8.0.33
- F* 2026.10.25 (Linux x86_64) with the Z3 build shipped in that release
- Host: Unix 6.12.113.0, Intel Xeon, 4 logical CPUs

## Tests

```text
$ dotnet exec bin/Release/net8.0/BootLegLM.dll test
PASS: matmul, causal softmax, shapes, prefix invariance, checkpoint roundtrip/digest/shape/nonfinite, BFS budgets, certificates, big integers, guided beam and best-first, ranking labels, bounded regex rewrite, multi-head stack, gradients, training, v3 checkpoints, roff theorems
```

Release build: 0 warnings, 0 errors.

## Training on examples/corpus.txt

Seed 42, 40 steps, sequence 12, learning rate 0.2, 2 layers, 4 heads, dim 16, hidden 32, vocab 128.

```text
loss_step=0,loss=4.806605
loss_step=39,loss=0.504180
saved=...,loss_initial=4.806605,loss_final=0.504180
```

The full series is `docs/train-corpus-loss.txt`. Steps 17, 37, and 39 rise because those windows contain a newline. `load` of that checkpoint on `abc` printed `Next token: 97`.

A second training from the same seed produced the same endpoints. Finite-difference checks on a smaller network passed inside `test` (absolute tolerance `1e-3`).

## Benchmarks

Five repetitions. Checksums did not change between repetitions. Full lines: `docs/bench-runs.txt`.

| Workload | min ms | median ms | max ms | checksum |
| --- | ---: | ---: | ---: | --- |
| 100 stack forwards | 18.683700 | 32.722100 | 34.317000 | -10.314740265845385 |
| 200 reason calls | 1.868100 | 1.950200 | 9.635300 | 400 |
| 20 SGD steps on `abc` | 20.082100 | 20.511600 | 29.176500 | final loss 0.029382611825437509 |

Legacy version-2 forward, one run inside the same command: `total_ms=68.387100`, checksum `-8.4242520795687188`.

## F*

```text
$ ./verify.sh
Verified module: RewriteSpec
Verified module: MatrixSpec
Verified module: ProofState
Verified module: NumericModel
F* specifications machine-checked by installed F* toolchain.
```

`ProofState.replay_bounded` shows a valid certificate has length at most the AST size of its start expression. `NumericModel.mean_of_constant` is a real-number identity. Neither result is an extraction of the F# implementation, and neither is an IEEE-754 proof.

## Still open

- No proof that F# `float` matmul, softmax, or layer-norm matches the real model.
- No extracted proof of the F# search queue.
- Regex `.ds` rules are not theorems.
- Training is single-window SGD. There is no Adam and no multi-sequence batch.
