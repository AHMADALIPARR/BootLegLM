# BootLegLM

CPU-only F# experimental transformer and deterministic symbolic rewriter.

Checkpoint format with a SHA-256 digest and a size cap, structural rewrite certificates, neural-ordered search rechecked by an independent certificate checker, F* specifications, and a small benchmark. Weights are deterministic random. The model is one causal attention head, a residual connection, and a ReLU feed-forward block. It has no positional encoding, layer norm, or training loop.

## License

Licensed under the [Business Source License 1.1](LICENSE). Non-commercial research and internal use is permitted. Commercial production use (hosted service or competing product) requires a separate license. Converts to Apache-2.0 on 2030-11-06.

## Requirements

- .NET 8 SDK, for the executable, `test`, and `bench`
- Optional: F* and the Z3 bundled with it, for `./verify.sh`

`docs/STATUS.md` records the toolchain versions and command output from the latest check. To install the SDK on Linux or macOS when `dotnet` is missing:

```sh
curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 8.0
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$DOTNET_ROOT:$PATH"
```

## Commands

```sh
dotnet run --project BootLegLM.fsproj -c Release -- test
dotnet run --project BootLegLM.fsproj -c Release -- save weights.blm
dotnet run --project BootLegLM.fsproj -c Release -- load weights.blm HELLO
dotnet run --project BootLegLM.fsproj -c Release -- reason '(a+0)*(b*1)'
dotnet run --project BootLegLM.fsproj -c Release -- guided '(a+0)*(b*1)'
dotnet run --project BootLegLM.fsproj -c Release -- guided '(a+0)*(b*1)' --strategy best
dotnet run --project BootLegLM.fsproj -c Release -- traindata
dotnet run --project BootLegLM.fsproj -c Release -- bench
dotnet run --project BootLegLM.fsproj -c Release -- rewrite examples/rules.roff 'x + 0'
dotnet run --project BootLegLM.fsproj -c Release -- help
dotnet run --project BootLegLM.fsproj -c Release -- HELLO

./verify.sh
# Override the binary: FSTAR=/path/to/fstar.exe ./verify.sh
```

Exit codes are 0 for success, 1 for an error, and 2 when a bounded search stops with `UNKNOWN`.

`test`, `save`, `load`, `reason`, `guided`, `traindata`, `bench`, `rewrite`, and `help` are commands. Any other argument is a generation prompt. `reason` accepts `--nodes N` (default 10000). `guided` accepts `--strategy beam|best` (default `beam`), `--beam N` (default 4, ignored by `best`), and `--nodes N` (default 1000).

The default network is vocabulary 128, dimension 16, feed-forward width 32, seed 42. Prompt characters become token ids by taking each UTF-16 code unit modulo 128. Direct calls to `forward` still reject ids outside `0 .. vocab-1`.

## Features

1. **Checkpoint** (BLM1 v2). Versioned little-endian payload, eight matrices, dimension and finite-value checks, temporary-file replace, SHA-256 over the payload, and a 50 MiB cap. Digest mismatches, bad headers, oversized matrices, and trailing payload bytes are rejected.
2. **Reasoning**. Integer AST with `+`, `*`, parentheses, variables, and nonnegative arbitrary-precision constants. Eight simplifying rules. Deterministic breadth-first search with a node budget and duplicate hashing. Each returned trace is replayed by `Reasoning.verifyTrace`.
3. **Guided search**. Beam search and best-first search. A single logit from the fixed-seed transformer orders candidates only. Equal scores break ties by the printed expression. Duplicate expressions are stored once. A normal form that has already been generated is still examined when the node budget is full. The certificate checker does not read the scores.
4. **Ranking proxy**. `traindata` marks successor rules as positive and adds one rule index that does not apply, then prints the mean squared error of `sum(features)` against that target. It does not change weights.
5. **F* specifications**. Integer identities, a total integer dot product, the root proof-state step, and contextual certificate replay (`ProofState.applies` and `ProofState.replay`) with an evaluation-soundness lemma. These do not model F# floating-point arithmetic, and the F# program is not extracted from F*. `./verify.sh` checks `RewriteSpec`, `MatrixSpec`, and `ProofState`.
6. **Benchmark**. 20 warmups and 200 timed 8-token forward passes, with runtime metadata and a checksum. See `BENCHMARKS.md`.
7. **Regex roff front-end**. Lines of the form `.ds NAME PATTERN => REPLACEMENT`. The rewriter stops after 64 passes, or before a replacement would exceed 8192 characters. This is separate from `reason` and `guided` and is not formally proved.

## Checkpoint layout

The file is a little-endian payload followed by the 32-byte SHA-256 of that payload.

- `int32` magic `0x424C4D31` (the characters `BLM1` read as a little-endian integer; the on-disk bytes are `31 4D 4C 42`)
- `int32` version `2`
- `int32` matrix count `8`
- eight matrices, each stored as `int32` rows, `int32` columns, then row-major `float64` values, in the order Emb, Wq, Wk, Wv, Wo, W1, W2, Head

A matrix may not exceed 10,000,000 elements. Save validates shapes and finiteness, writes `path.tmp`, flushes it, and then replaces `path`.

## Boundaries

- The network is a small experimental block with random weights. It is not a trained language model.
- Regex rewriting is not the certificate-checked structural rewriter. `0 + x` is simplified by `reason` and left unchanged by `examples/rules.roff`.
- Certificates stay in memory. They are not exported as a standard proof object.
- `UNKNOWN` means the node budget ended before a normal form. Search termination is not machine-checked.
- Multi-head attention, layer norm, a multi-layer stack, gradient training, a floating-point F* model of the F# runtime, and roff theorem declarations beyond `.ds` rules are not implemented.
