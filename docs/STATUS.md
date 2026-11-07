# BootLegLM — Evidence Status

Checked in this workspace on 2026-11-07.

## Toolchains

- .NET SDK 8.0.427
- Microsoft.NETCore.App 8.0.33
- F* 2026.10.25 (Linux x86_64), with the Z3 build shipped in that release
- Host reported by the benchmark: `Unix 6.12.113.0`

## Correctness

```text
$ dotnet run --project BootLegLM.fsproj -c Release --no-build -- test
PASS: matmul, causal softmax, shapes, prefix invariance, checkpoint roundtrip/digest/shape/nonfinite, BFS budgets, certificates, big integers, guided beam and best-first, ranking labels, bounded regex rewrite
```

Release build of `BootLegLM.fsproj` completed with 0 warnings and 0 errors (`TreatWarningsAsErrors` is enabled).

## Symbolic and guided commands

Breadth-first `reason` follows successor order:

```text
$ dotnet exec bin/Release/net8.0/BootLegLM.dll reason '(a+0)*(b*1)'
AddZeroR: ((a+0)*(b*1)) -> (a*(b*1))
MulOneR: (a*(b*1)) -> (a*b)
CERTIFICATE: verified 2 steps
RESULT: (a*b)
```

The two one-step rewrites of this input have the same feature vector, so the transformer scores tie. Beam and best-first break that tie by the printed expression. Both still return a certificate that `Reasoning.verifyTrace` accepts:

```text
$ dotnet exec bin/Release/net8.0/BootLegLM.dll guided '(a+0)*(b*1)'
MulOneR: ((a+0)*(b*1)) -> ((a+0)*b)
AddZeroR: ((a+0)*b) -> (a*b)
CERTIFICATE: verified 2 steps (nodes=4)
RESULT: (a*b)

$ dotnet exec bin/Release/net8.0/BootLegLM.dll guided '(a+0)*(b*1)' --strategy best
MulOneR: ((a+0)*(b*1)) -> ((a+0)*b)
AddZeroR: ((a+0)*b) -> (a*b)
CERTIFICATE: verified 2 steps (nodes=4)
RESULT: (a*b)
```

A budget of 3 nodes stops before the normal form (exit code 2). A budget of 4 nodes finds it. `load` of a seed-42 checkpoint on `HELLO` prints `Next token: 42`. Flipping the last digest byte is rejected with `checkpoint digest mismatch`.

## Ranking proxy

The previous recorded value `ranking_mse=124.363636` labeled rule index 0 as valid when `AddZeroL` did not apply, and as invalid when it did. The labels now follow the successor relation. On the same five expressions the measured proxy is:

```text
$ dotnet exec bin/Release/net8.0/BootLegLM.dll traindata
examples=11 ranking_mse=132.545455
```

This is a mean squared error of a linear feature sum. The transformer weights are not updated.

## Benchmark

Five separate process runs. The checksum matches across runs. The timings are wall-clock observations on this host, not a hardware comparison.

```text
benchmark,os=Unix 6.12.113.0,runtime=.NET 8.0.33,iterations=200,seq=8,vocab=128,dim=16,hidden=32,total_ms=72.105700,ms_per_forward=0.360528,checksum=-8.4242520795687188
benchmark,os=Unix 6.12.113.0,runtime=.NET 8.0.33,iterations=200,seq=8,vocab=128,dim=16,hidden=32,total_ms=69.363600,ms_per_forward=0.346818,checksum=-8.4242520795687188
benchmark,os=Unix 6.12.113.0,runtime=.NET 8.0.33,iterations=200,seq=8,vocab=128,dim=16,hidden=32,total_ms=72.783300,ms_per_forward=0.363916,checksum=-8.4242520795687188
benchmark,os=Unix 6.12.113.0,runtime=.NET 8.0.33,iterations=200,seq=8,vocab=128,dim=16,hidden=32,total_ms=72.633300,ms_per_forward=0.363167,checksum=-8.4242520795687188
benchmark,os=Unix 6.12.113.0,runtime=.NET 8.0.33,iterations=200,seq=8,vocab=128,dim=16,hidden=32,total_ms=69.437500,ms_per_forward=0.347187,checksum=-8.4242520795687188
```

## F*

```text
$ ./verify.sh
Verified module: RewriteSpec
All verification conditions discharged successfully
Verified module: MatrixSpec
All verification conditions discharged successfully
Verified module: ProofState
All verification conditions discharged successfully
F* specifications machine-checked by installed F* toolchain.
```

`ProofState.applies` and `ProofState.replay` are the contextual certificate checker. The development includes checked samples for the two-step trace, a root step, a fold under multiplication, a wrong rule, and a tampered step. Search termination is not proved.

## Evidence table

| Component            | Implemented | Compiled | Tested | Formally verified |
|----------------------|-------------|----------|--------|-------------------|
| Symbolic reasoning   | Yes         | Yes      | Yes    | Yes (integer lemmas, root kernel, contextual replay) |
| Matrix algorithms    | Yes (float runtime, integer spec) | Yes | Yes | Yes (integer dot / mm_entry only) |
| Transformer          | Partial (single head, random weights) | Yes | Yes | No |
| Checkpoint system    | Yes         | Yes      | Yes    | No |
| Neural-guided search | Yes (beam and best-first) | Yes | Yes | No (ordering only; replay kernel verified) |
| Ranking proxy        | Yes (measured MSE, no weight update) | Yes | Yes | No |
| Regex roff front-end | Yes (pass and size caps) | Yes | Yes | No |

## Remaining

- Gradient-based training of the proposal model. `traindata` does not backpropagate.
- Multi-head attention, layer norm, and a multi-layer stack.
- An F* floating-point model related to the F# runtime.
- A machine-checked termination argument for bounded search.
- Roff theorem declarations beyond `.ds` pattern rules.
