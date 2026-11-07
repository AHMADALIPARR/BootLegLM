# Reproducible evaluation protocol

Measured numbers live in [`docs/bench-runs.txt`](docs/bench-runs.txt) and [`docs/train-corpus-loss.txt`](docs/train-corpus-loss.txt). Replace those files only with new command output, then run `python3 tools/render_docs.py`. Do not invent timings.

## Correctness

```sh
dotnet --info
dotnet run --project BootLegLM.fsproj -c Release -- test
```

`test` includes the legacy single-head checks, multi-head causal shapes, finite-difference gradients, a 40-step training run on `examples/corpus.txt` that must finish at least 0.5 below its first loss and match a second run, version-2 and version-3 checkpoints, and `.thm` proofs.

## Wall-clock suite

```sh
dotnet run --project BootLegLM.fsproj -c Release -- bench
```

The suite prints the OS, CPU model, logical CPU count, and runtime, then five repetitions of:

1. **Inference.** Default stack, seed 42, vocab 128, dim 16, 4 heads, 2 layers, hidden 32. Ten warmup forwards and 100 timed forwards of tokens `65..72`. Checksum is the sum of `logits[7,0]`.
2. **Search.** Two hundred calls to breadth-first `reason` on `(a+0)*(b*1)` with a node budget of 1000. Checksum is the total number of certificate steps.
3. **Training.** A fresh network each repetition. Twenty SGD steps, sequence length 12, learning rate 0.2, seed 42, on 240 characters of `abc` (no newlines). Checksum is the final loss.

It then prints one legacy version-2 forward benchmark: 20 warmup and 200 timed forwards. That checksum is the historical single-head value.

Report every repetition. The first repetition can include JIT time. `ms` values are wall time for the whole workload, not per token and not a comparison with another system.

## Corpus training curve

```sh
dotnet run --project BootLegLM.fsproj -c Release -- train examples/corpus.txt weights.blm
```

Defaults: 40 steps, learning rate 0.2, sequence 12, seed 42. The window starts at `(step * 3) mod (length - 12)`. Windows that contain a newline are a different distribution from pure `abc`, so the loss is not monotone. The recorded curve is `docs/train-corpus-loss.txt`.

## F*

```sh
./verify.sh
```

Checks `RewriteSpec`, `MatrixSpec`, `ProofState` (including the certificate size bound), and `NumericModel`. The real-number mean is not an IEEE-754 proof.
