(* Copyright (c) 2026 Ahmad Ali Parr / SnapKitty Research Lab
   Licensed under the Business Source License 1.1.
   See LICENSE in the repository root. *)

module NumericModel

(* Real-number model of two facts used by the F# stack:
   head splits tile the model dimension, and a causal mask drops the future.
   This is not an IEEE-754 proof and it is not extracted to System.Double.
   Bit-exact correspondence with F# float64 matmul is not claimed. *)

let head_split (dim:nat) (heads:nat)
  : Lemma (requires (heads > 0 /\ dim > 0 /\ dim % heads = 0))
          (ensures (heads * (dim / heads) = dim)) = ()

let causal_visible (i:nat) (j:nat) : Tot bool = j <= i

let future_blocked (i:nat) (j:nat)
  : Lemma (requires (j > i))
          (ensures (causal_visible i j = false)) = ()

let past_visible (i:nat) (j:nat)
  : Lemma (requires (j <= i))
          (ensures (causal_visible i j = true)) = ()

open FStar.Real

(* Population sum of n copies of a real. The mean of a constant vector is that constant.
   Layer-norm in the F# stack uses this population mean. The identity is over
   mathematical reals, not System.Double. *)
let rec sum_const (n:nat) (c:real) : Tot real (decreases n) =
  if n = 0 then 0.0R else c +. sum_const (n - 1) c

let rec sum_const_closed (n:nat) (c:real)
  : Lemma (ensures (sum_const n c == of_int n *. c))
          (decreases n) =
  if n = 0 then () else sum_const_closed (n - 1) c

let of_int_nonzero (n:pos)
  : Lemma (ensures (of_int n =!= 0.0R)) = ()

let mean_of_constant (n:pos) (c:real)
  : Lemma (ensures (sum_const n c /. of_int n == c)) =
  of_int_nonzero n;
  sum_const_closed n c
