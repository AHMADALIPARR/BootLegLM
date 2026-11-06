# Reproducible evaluation protocol

No benchmark figures are claimed in this repository: an executing .NET 8 toolchain is needed.

## Correctness (run first)

```sh
dotnet --info
dotnet run --project BootLegLM.fsproj -c Release -- test
```

Tests check matrix entries, causal masking/row sums, output shape, causal prefix invariance, byte-exact checkpoint/logit roundtrip, rejection of trailing checkpoint bytes, basic breadth-first rewriting, arithmetic equivalence over sample assignments, certificate replay and tampering rejection, and arbitrary precision constant folding.

## Performance

```sh
dotnet run --project BootLegLM.fsproj -c Release --no-launch-profile -- bench
```

The benchmark performs 20 warmup and 200 timed forward evaluations of an 8-token context (vocab=128, dim=16, FFN=32). Output contains the wall-clock duration and a checksum to prevent mistaking mere loop overhead for work. Capture CPU, operating system, `dotnet --info`, compiler configuration, and at least five independent process runs. Do not interpret `ms_per_forward` as autoregressive token latency or claim hardware speedups from it.

## Reasoning/search

```sh
dotnet run --project BootLegLM.fsproj -c Release -- reason '(a+0)*(b*1)'
```

A deterministic bounded BFS explores a **small fixed set** of arithmetic identities and returns a trace replayed by an independent validation pass. `UNKNOWN` means the bounded search did not find a normal form; no theorem proving completeness is claimed. Arithmetic expressions use arbitrary-precision mathematical integers, matching the domain of the F* lemmas.

## F* formal checks

```sh
./verify.sh
# Override binary: FSTAR=/path/to/fstar.exe ./verify.sh
```

This script exits on the first failed proof. The F* toolchain is absent in the creation environment, so these files are **not reported as machine-checked**. They specify mathematical integer identities and matrix indexing, not the machine-level semantics of F# floating-point matmul. That correspondence remains open.
