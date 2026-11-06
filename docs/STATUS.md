# BootLegLM — Evidence Status

Toolchain check performed in the build environment on 2026-11-06:

- `dotnet` : not found
- `fstar` / `fstar.exe` : not found
- Z3 : not found

No compilation, test, or F* typechecking results are claimed. All entries below are therefore "No" for Compiled / Tested / Formally verified until the corresponding commands succeed on a machine that has the toolchains.

| Component            | Implemented | Compiled | Tested | Formally verified |
|----------------------|-------------|----------|--------|-------------------|
| Symbolic reasoning   | Partial     | No       | No     | No                |
| Matrix algorithms    | Partial     | No       | No     | No                |
| Transformer          | Partial     | No       | No     | No                |
| Checkpoint system    | Yes         | No       | No     | No                |
| Training             | No          | No       | No     | No                |

## What exists

- F# (`src/Program.fs`): Tensor ops, single-head causal attention + ReLU FFN, BLM1 v2 checkpoint with SHA-256 and size cap, structural integer AST rewriter with BFS and certificate replay, regex roff front-end, unit checks, benchmark harness.
- F* (`proofs/`): integer rewrite identities and a total recursive integer dot-product / matrix entry. These are obligations, not yet machine-checked.
- Checkpoint: versioned binary, magic, version, dimensions, payload, SHA-256 digest, atomic rename, size cap, trailing-byte and digest rejection.

## What is not yet present

- Multi-head attention, layer norm, multi-layer stack (beyond single block).
- Training loop, cross-entropy, backprop, AdamW, gradient checks.
- Full F* executable reasoning kernel with measure-based termination proof and independent certificate checker.
- F* floating-point numerical model and proofs relating it to the F# runtime.
- Actual F* verification logs or `dotnet test` output.

## Commands that must succeed before any "Yes" is recorded

```sh
# F# (requires .NET 8 SDK)
dotnet run --project BootLegLM.fsproj -c Release -- test
dotnet run --project BootLegLM.fsproj -c Release -- bench

# F* (requires F* + Z3)
./verify.sh
```

Until those commands produce successful exit codes and logs, this table remains as written.
