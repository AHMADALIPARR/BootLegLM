# BootLegLM

CPU-only F# experimental transformer and deterministic symbolic rewriter.

Production-oriented checkpoint format with SHA-256 digest and size cap, structural rewrite certificates, F* mathematical specifications, and benchmarks. Not trained. Weights are deterministic random. No positional encoding or multi-head attention beyond the single-head causal block currently implemented.

## License

Licensed under the [Business Source License 1.1](LICENSE). Non-commercial research and internal use is permitted. Commercial production use (hosted service or competing product) requires a separate license. Converts to Apache-2.0 on 2030-11-06.

## Requirements

- .NET 8 SDK (for the F# executable, tests, and benchmarks)
- Optional: F* + Z3 (for machine-checking the specifications in `proofs/`)

Neither toolchain is present in the environment that produced the current commit. See `docs/STATUS.md` for the evidence table.

## Commands

```sh
# Correctness (requires .NET 8)
dotnet run --project BootLegLM.fsproj -c Release -- test

# Checkpoint
dotnet run --project BootLegLM.fsproj -c Release -- save weights.blm
dotnet run --project BootLegLM.fsproj -c Release -- load weights.blm HELLO

# Symbolic reasoning
dotnet run --project BootLegLM.fsproj -c Release -- reason '(a+0)*(b*1)'

# Benchmark (requires .NET 8; do not invent numbers)
dotnet run --project BootLegLM.fsproj -c Release -- bench

# Regex roff front-end
dotnet run --project BootLegLM.fsproj -c Release -- rewrite examples/rules.roff 'x + 0'

# F* (requires F* + Z3)
./verify.sh
```

## Features

1. **Checkpoint** (BLM1 v2): versioned binary, eight matrices, dimension and finite-value checks, temporary-file atomic save, SHA-256 digest over the payload, hard file-size cap. Trailing data and digest mismatch are rejected.
2. **Reasoning**: integer AST with `+`, `*`, parentheses and variables; eight built-in simplifying rules; deterministic breadth-first bounded search; independently replayed per-step certificate. Arbitrary-precision integers match the F* domain.
3. **F* specifications**: mathematical integer identities and a total recursive integer dot-product / matrix entry in `proofs/`. These do not verify the F# floating-point implementation and have not been machine-checked in this environment.
4. **Benchmark**: 20 warmups + 200 timed 8-token forward passes with runtime metadata and checksum. See `BENCHMARKS.md`.

## Boundaries

- Single-head causal attention + ReLU feed-forward. Random weights only; no training or back-propagation.
- Regex roff-style rewrite is distinct from the certificate-checked structural `reason` command and is not formally proven.
- Certificates are internal; they are not exported in a standard proof format.
- The transformer is a small experimental model, not a production language model.
- Full multi-head attention, layer norm, multi-layer stack, and a training loop are not yet implemented.
- See `docs/STATUS.md` for the component evidence table. No compilation or verification is claimed until the corresponding commands succeed.
