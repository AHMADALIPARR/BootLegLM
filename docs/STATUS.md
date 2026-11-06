# BootLegLM — Evidence Status

Toolchain provisioned and verified in this environment on 2026-11-06.

## Toolchains

- .NET SDK 8.0.427 (installed via official tarball, --no-same-owner)
- F* 2026.10.25 (Linux x86_64 binary)
- Z3 5.3.0 (bundled with F*)

## Commands executed and results

```text
$ dotnet exec bin/Release/net8.0/BootLegLM.dll test
PASS: matmul, causal softmax, shapes, prefix invariance, checkpoint roundtrip/rejection, BFS rewriting, checked certificates, big integers

$ fstar proofs/RewriteSpec.fst
Verified module: RewriteSpec
All verification conditions discharged successfully

$ fstar proofs/MatrixSpec.fst
Verified module: MatrixSpec
All verification conditions discharged successfully

$ dotnet exec bin/Release/net8.0/BootLegLM.dll reason '(a+0)*(b*1)'
AddZeroR: ((a+0)*(b*1)) -> (a*(b*1))
MulOneR: (a*(b*1)) -> (a*b)
CERTIFICATE: verified 2 steps
RESULT: (a*b)

$ dotnet exec bin/Release/net8.0/BootLegLM.dll bench
benchmark,os=Unix 6.12.8.0,runtime=.NET 8.0.33,iterations=200,seq=8,vocab=128,dim=16,hidden=32,total_ms=99.417000,ms_per_forward=0.497085,checksum=-8.4242520795687188
```

Note: the standalone executable bit could not be set (sandbox restriction); tests were run via `dotnet exec` on the DLL. `verify.sh` shebang failed with "bad interpreter" (same restriction); F* was invoked directly.

## Evidence table

| Component            | Implemented | Compiled | Tested | Formally verified |
|----------------------|-------------|----------|--------|-------------------|
| Symbolic reasoning   | Partial     | Yes      | Yes    | Yes (integer lemmas) |
| Matrix algorithms    | Partial     | Yes      | Yes    | Yes (integer dot / mm_entry) |
| Transformer          | Partial     | Yes      | Yes    | No                |
| Checkpoint system    | Yes         | Yes      | Yes    | No                |
| Training             | No          | No       | No     | No                |

## Remaining

- Multi-head attention, layer norm, multi-layer stack not implemented.
- Training loop, backprop, AdamW not implemented.
- F* floating-point numerical model and proofs relating it to F# runtime not present.
- Full measure-based termination proof for the search not machine-checked beyond the existing integer identities.
