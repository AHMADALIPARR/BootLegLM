(* Copyright (c) 2026 Ahmad Ali Parr / SnapKitty Research Lab
   Licensed under the Business Source License 1.1.
   See LICENSE in the repository root. *)

module RewriteSpec

// Total, pure integer identities. Floating-point and machine-overflow
// properties are not implied by these lemmas.
let add_zero_right (x:int) : Lemma (ensures (x + 0 = x)) = ()
let add_zero_left (x:int) : Lemma (ensures (0 + x = x)) = ()
let mul_one_right (x:int) : Lemma (ensures (x * 1 = x)) = ()
let mul_one_left (x:int) : Lemma (ensures (1 * x = x)) = ()
let mul_zero_right (x:int) : Lemma (ensures (x * 0 = 0)) = ()
let mul_zero_left (x:int) : Lemma (ensures (0 * x = 0)) = ()
let fold_add (x:int) (y:int) : Lemma (ensures (x+y = y+x)) = ()
