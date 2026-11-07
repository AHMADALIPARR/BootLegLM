#!/usr/bin/env sh
# Copyright (c) 2026 Ahmad Ali Parr / SnapKitty Research Lab
# Licensed under the Business Source License 1.1. See LICENSE.
set -eu
cd "$(dirname "$0")"
: "${FSTAR:=fstar.exe}"
"$FSTAR" proofs/RewriteSpec.fst
"$FSTAR" proofs/MatrixSpec.fst
"$FSTAR" proofs/ProofState.fst
"$FSTAR" proofs/NumericModel.fst
printf '%s\n' 'F* specifications machine-checked by installed F* toolchain.'
