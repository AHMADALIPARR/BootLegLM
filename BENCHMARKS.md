# Reproducible evaluation protocol

Do not copy timings from this file into a performance claim. Wall-clock numbers belong in `docs/STATUS.md` only after the commands below have been run, and they describe that machine.

## Correctness (run first)

```sh
dotnet --info
dotnet run --project BootLegLM.fsproj -c Release -- test
```

`test` checks matrix entries, causal masking and row sums, output shape, causal prefix invariance, argmax, deterministic token encoding, checkpoint roundtrip, digest and header rejection, non-finite and shape rejection, breadth-first rewriting, node budgets, arithmetic equivalence, certificate replay and tampering, arbitrary-precision folding, beam and best-first search, ranking labels, and the regex pass and size caps.

## Performance

```sh
dotnet run --project BootLegLM.fsproj -c Release --no-launch-profile -- bench
```

The benchmark performs 20 warmup and 200 timed forward evaluations of an 8-token context (vocab=128, dim=16, FFN=32). The line contains the wall-clock duration and a checksum so a no-op loop cannot be mistaken for the work. Capture CPU, operating system, `dotnet --info`, compiler configuration, and at least five independent process runs. `ms_per_forward` is the measured forward time for this fixed context. It is not autoregressive token latency.

## Reasoning and guided search

```sh
dotnet run --project BootLegLM.fsproj -c Release -- reason '(a+0)*(b*1)'
dotnet run --project BootLegLM.fsproj -c Release -- guided '(a+0)*(b*1)'
dotnet run --project BootLegLM.fsproj -c Release -- guided '(a+0)*(b*1)' --strategy best
dotnet run --project BootLegLM.fsproj -c Release -- traindata
```

`reason` is a deterministic bounded BFS over a fixed set of arithmetic identities. `guided` orders the same rules with the frozen transformer. Both return a trace that an independent checker replays. `UNKNOWN` and exit code 2 mean the node budget ended first. No completeness is claimed. `traindata` prints a ranking mean squared error. It does not update weights.

## Regex front-end

```sh
dotnet run --project BootLegLM.fsproj -c Release -- rewrite examples/rules.roff 'x + 0'
```

This is a bounded regular-expression pass, not the certificate checker.

## F* formal checks

```sh
./verify.sh
# Override binary: FSTAR=/path/to/fstar.exe ./verify.sh
```

The script checks `proofs/RewriteSpec.fst`, `proofs/MatrixSpec.fst`, and `proofs/ProofState.fst`, and it exits on the first failure. The specifications are mathematical integers and certificate replay. They do not describe IEEE-754 floating-point matmul in the F# runtime.
