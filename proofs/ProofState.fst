(* Copyright (c) 2026 Ahmad Ali Parr / SnapKitty Research Lab
   Licensed under the Business Source License 1.1.
   See LICENSE in the repository root. *)

module ProofState

(* Trusted proof-state kernel. Neural scores are not part of this module.
   local_step and history_sound cover a root redex.
   applies and replay cover the one-hole contextual closure checked by
   the F# certificate checker. Search termination is not claimed. *)

type expr =
  | EConst of int
  | EVar of string
  | EAdd of expr & expr
  | EMul of expr & expr

type rule =
  | AddZeroL
  | AddZeroR
  | MulOneL
  | MulOneR
  | MulZeroL
  | MulZeroR
  | FoldAdd
  | FoldMul

type step = {
  rule: rule;
  before: expr;
  after: expr
}

type state = {
  goal: expr;
  assumptions: list expr;
  subgoals: list expr;
  depth: nat;
  max_depth: nat;
  history: list step
}

let rec eval (e:expr) (env: string -> int) : Tot int (decreases e) =
  match e with
  | EConst n -> n
  | EVar s -> env s
  | EAdd (a, b) -> eval a env + eval b env
  | EMul (a, b) -> eval a env * eval b env

let local_step (e:expr) : Tot (option (rule & expr)) =
  match e with
  | EAdd (EConst 0, x) -> Some (AddZeroL, x)
  | EAdd (x, EConst 0) -> Some (AddZeroR, x)
  | EMul (EConst 1, x) -> Some (MulOneL, x)
  | EMul (x, EConst 1) -> Some (MulOneR, x)
  | EMul (EConst 0, _) -> Some (MulZeroL, EConst 0)
  | EMul (_, EConst 0) -> Some (MulZeroR, EConst 0)
  | EAdd (EConst a, EConst b) -> Some (FoldAdd, EConst (a + b))
  | EMul (EConst a, EConst b) -> Some (FoldMul, EConst (a * b))
  | _ -> None

let local_sound (e:expr) (r:rule) (e2:expr) (env: string -> int)
  : Lemma (requires (local_step e = Some (r, e2)))
          (ensures (eval e env = eval e2 env)) = ()

let rec history_sound (h:list step) (env: string -> int) : Tot bool (decreases h) =
  match h with
  | [] -> true
  | s :: rest ->
    (match local_step s.before with
     | Some (r, e2) -> r = s.rule && e2 = s.after && eval s.before env = eval s.after env
     | None -> false)
    && history_sound rest env

let state_valid (s:state) (env: string -> int) : Tot bool =
  history_sound s.history env

let step_preserves (s:state) (r:rule) (e2:expr) (env: string -> int)
  : Lemma
    (requires (local_step s.goal = Some (r, e2) && state_valid s env))
    (ensures (state_valid
               { s with
                 goal = e2;
                 depth = s.depth + 1;
                 history = { rule = r; before = s.goal; after = e2 } :: s.history }
               env)) = ()

(* One-hole contextual closure of local_step, matching F# successors:
   a step rewrites the root or exactly one child, recursively. *)
let rec applies (e:expr) (r:rule) (e2:expr) : Tot bool (decreases e) =
  (match local_step e with
   | Some (r', e') -> r = r' && e2 = e'
   | None -> false)
  || (match e, e2 with
      | EAdd (a, b), EAdd (a', b') ->
        (b = b' && applies a r a') || (a = a' && applies b r b')
      | EMul (a, b), EMul (a', b') ->
        (b = b' && applies a r a') || (a = a' && applies b r b')
      | _ -> false)

let rec applies_sound (e:expr) (r:rule) (e2:expr) (env:string -> int)
  : Lemma (requires (applies e r e2))
          (ensures (eval e env == eval e2 env))
          (decreases e) =
  match e with
  | EConst _ | EVar _ -> ()
  | EAdd (a, b) ->
    (match local_step e with
     | Some (r', e') ->
       if r = r' && e2 = e' then local_sound e r e2 env
       else ()
     | None -> ());
    (match e2 with
     | EAdd (a', b') ->
       if b = b' && applies a r a' then applies_sound a r a' env
       else if a = a' && applies b r b' then applies_sound b r b' env
       else ()
     | _ -> ())
  | EMul (a, b) ->
    (match local_step e with
     | Some (r', e') ->
       if r = r' && e2 = e' then local_sound e r e2 env
       else ()
     | None -> ());
    (match e2 with
     | EMul (a', b') ->
       if b = b' && applies a r a' then applies_sound a r a' env
       else if a = a' && applies b r b' then applies_sound b r b' env
       else ()
     | _ -> ())

let rec replay (src:expr) (dst:expr) (steps:list step) : Tot bool (decreases steps) =
  match steps with
  | [] -> src = dst
  | s :: rest ->
    s.before = src && applies src s.rule s.after && replay s.after dst rest

let rec replay_sound (src:expr) (dst:expr) (steps:list step) (env:string -> int)
  : Lemma (requires (replay src dst steps))
          (ensures (eval src env == eval dst env))
          (decreases steps) =
  match steps with
  | [] -> ()
  | s :: rest ->
    applies_sound src s.rule s.after env;
    replay_sound s.after dst rest env

let sample_add_zero : step = {
  rule = AddZeroR;
  before = EMul (EAdd (EVar "a", EConst 0), EMul (EVar "b", EConst 1));
  after = EMul (EVar "a", EMul (EVar "b", EConst 1))
}

let sample_mul_one : step = {
  rule = MulOneR;
  before = sample_add_zero.after;
  after = EMul (EVar "a", EVar "b")
}

let sample_trace_sound (env:string -> int)
  : Lemma (ensures (
      replay sample_add_zero.before sample_mul_one.after [sample_add_zero; sample_mul_one] /\
      eval sample_add_zero.before env = eval sample_mul_one.after env)) =
  replay_sound sample_add_zero.before sample_mul_one.after [sample_add_zero; sample_mul_one] env

let sample_root_sound (env:string -> int)
  : Lemma (ensures (
      applies (EAdd (EVar "x", EConst 0)) AddZeroR (EVar "x") /\
      eval (EAdd (EVar "x", EConst 0)) env = eval (EVar "x") env)) =
  applies_sound (EAdd (EVar "x", EConst 0)) AddZeroR (EVar "x") env

let sample_fold_inside_sound (env:string -> int)
  : Lemma (ensures (
      applies (EMul (EAdd (EConst 2, EConst 3), EVar "z")) FoldAdd (EMul (EConst 5, EVar "z")) /\
      eval (EMul (EAdd (EConst 2, EConst 3), EVar "z")) env =
        eval (EMul (EConst 5, EVar "z")) env)) =
  applies_sound (EMul (EAdd (EConst 2, EConst 3), EVar "z")) FoldAdd (EMul (EConst 5, EVar "z")) env

let sample_wrong_rule_rejected ()
  : Lemma (ensures (applies (EAdd (EVar "a", EConst 0)) AddZeroL (EVar "a") = false)) = ()

let sample_tampered_rejected ()
  : Lemma (ensures (
      replay sample_add_zero.before (EConst 999)
        [{ sample_add_zero with after = EConst 999 }] = false)) = ()

(* Every accepted contextual step strictly decreases AST size, so a
   certificate cannot be an infinite rewrite chain. The bound is on the
   certificate, not an extraction of the F# search queue. *)
let rec size (e:expr) : Tot nat (decreases e) =
  match e with
  | EConst _ | EVar _ -> 1
  | EAdd (a, b) | EMul (a, b) -> 1 + size a + size b

let local_decreases (e:expr) (r:rule) (e2:expr)
  : Lemma (requires (local_step e = Some (r, e2)))
          (ensures (size e2 < size e)) = ()

let rec applies_decreases (e:expr) (r:rule) (e2:expr)
  : Lemma (requires (applies e r e2))
          (ensures (size e2 < size e))
          (decreases e) =
  match e with
  | EConst _ | EVar _ -> ()
  | EAdd (a, b) ->
    (match local_step e with
     | Some (r', e') ->
       if r = r' && e2 = e' then local_decreases e r e2
       else ()
     | None -> ());
    (match e2 with
     | EAdd (a', b') ->
       if b = b' && applies a r a' then applies_decreases a r a'
       else if a = a' && applies b r b' then applies_decreases b r b'
       else ()
     | _ -> ())
  | EMul (a, b) ->
    (match local_step e with
     | Some (r', e') ->
       if r = r' && e2 = e' then local_decreases e r e2
       else ()
     | None -> ());
    (match e2 with
     | EMul (a', b') ->
       if b = b' && applies a r a' then applies_decreases a r a'
       else if a = a' && applies b r b' then applies_decreases b r b'
       else ()
     | _ -> ())

let rec replay_bounded (src:expr) (dst:expr) (steps:list step)
  : Lemma (requires (replay src dst steps))
          (ensures (List.Tot.length steps <= size src))
          (decreases steps) =
  match steps with
  | [] -> ()
  | s :: rest ->
    applies_decreases src s.rule s.after;
    replay_bounded s.after dst rest

let sample_trace_bounded ()
  : Lemma (ensures (List.Tot.length [sample_add_zero; sample_mul_one] <= size sample_add_zero.before)) =
  replay_bounded sample_add_zero.before sample_mul_one.after [sample_add_zero; sample_mul_one]
