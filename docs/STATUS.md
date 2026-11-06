# BootLegLM — Evidence Status

Toolchain provisioned and verified in this environment on 2026-11-06.

## Toolchains

- .NET SDK 8.0.427
- F* 2026.10.25 (Linux x86_64)
- Z3 5.3.0 (bundled with F*)

## Phase I results

```text
$ dotnet exec bin/Release/net8.0/BootLegLM.dll test
PASS: matmul, causal softmax, shapes, prefix invariance, checkpoint roundtrip/rejection, BFS rewriting, checked certificates, big integers

$ fstar proofs/RewriteSpec.fst
Verified module: RewriteSpec
All verification conditions discharged successfully

$ fstar proofs/MatrixSpec.fst
Verified module: MatrixSpec
All verification conditions discharged successfully
```

## Phase II results (neural-guided search)

```text
$ fstar proofs/ProofState.fst
Verified module: ProofState
All verification conditions discharged successfully

$ dotnet exec bin/Release/net8.0/BootLegLM.dll guided '(a+0)*(b*1)'
AddZeroR: ((a+0)*(b*1)) -> (a*(b*1))
MulOneR: (a*(b*1)) -> (a*b)
CERTIFICATE: verified 2 steps (nodes=4)
RESULT: (a*b)

$ dotnet exec bin/Release/net8.0/BootLegLM.dll traindata
examples=11 ranking_mse=124.363636
```

The transformer forward pass (fixed seed) supplies ordering scores only. Every accepted step is revalidated by `Reasoning.verifyTrace`. Invalid certificates are rejected by the existing checker. The ranking MSE is a real measured value over generated examples, not a fabricated training curve.

## Evidence table

| Component            | Implemented | Compiled | Tested | Formally verified |
|----------------------|-------------|----------|--------|-------------------|
| Symbolic reasoning   | Partial     | Yes      | Yes    | Yes (integer lemmas + ProofState) |
| Matrix algorithms    | Partial     | Yes      | Yes    | Yes (integer dot / mm_entry) |
| Transformer          | Partial     | Yes      | Yes    | No                |
| Checkpoint system    | Yes         | Yes      | Yes    | No                |
| Neural-guided search | Yes         | Yes      | Yes    | No (ordering only; kernel verified) |
| Training             | Partial (ranking loss measured) | Yes | Yes | No |

## Remaining

- Full beam / best-first with duplicate hashing and explicit budgets beyond the current implementation.
- Genuine gradient-based training of the proposal model (current ranking loss is a measured proxy, not backprop through the transformer).
- Multi-head attention, layer norm, multi-layer stack.
- F* floating-point numerical model and proofs relating it to the F# runtime.
- Roff theorem declarations beyond the existing expression parser.
