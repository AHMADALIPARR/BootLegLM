(* Copyright (c) 2026 Ahmad Ali Parr / SnapKitty Research Lab
   Licensed under the Business Source License 1.1.
   See LICENSE in the repository root. *)

module MatrixSpec

// Mathematical integer specification (not floating-point F# verification).
// An explicit structurally decreasing dot product avoids hidden library axioms.
let rec dot (n:nat) (a:nat -> int) (b:nat -> int)
  : Tot int (decreases n) =
  if n = 0 then 0
  else dot (n - 1) a b + a (n - 1) * b (n - 1)

let mm_entry (m:nat) (n:nat) (p:nat)
             (a:nat -> nat -> int) (b:nat -> nat -> int)
             (i:nat{i < m}) (j:nat{j < p}) : Tot int =
  dot n (a i) (fun k -> b k j)

let dot_empty (a:nat -> int) (b:nat -> int)
  : Lemma (ensures (dot 0 a b = 0)) = ()
