(* Copyright (c) 2026 Ahmad Ali Parr / SnapKitty Research Lab
   Licensed under the Business Source License 1.1.
   See LICENSE in the repository root. *)

module RewriteSpec

(* Integer identities used by the structural rewriter.
   Each lemma is discharged by the SMT solver (no admit).
   Floating-point and machine-overflow properties are not claimed. *)

let add_zero_right (x:int) : Lemma (ensures (x + 0 = x)) = ()
let add_zero_left (x:int) : Lemma (ensures (0 + x = x)) = ()
let mul_one_right (x:int) : Lemma (ensures (x * 1 = x)) = ()
let mul_one_left (x:int) : Lemma (ensures (1 * x = x)) = ()
let mul_zero_right (x:int) : Lemma (ensures (x * 0 = 0)) = ()
let mul_zero_left (x:int) : Lemma (ensures (0 * x = 0)) = ()
let fold_add (x:int) (y:int) : Lemma (ensures (x + y = y + x)) = ()
let fold_mul (x:int) (y:int) : Lemma (ensures (x * y = y * x)) = ()

(* Evaluation is the identity on integers; soundness of the rules
   follows immediately from the lemmas above for constant cases.
   A full AST interpreter and measure-based termination proof
   for the search are not yet machine-checked in this file. *)
