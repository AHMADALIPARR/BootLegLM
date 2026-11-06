(* Copyright (c) 2026 Ahmad Ali Parr / SnapKitty Research Lab
   Licensed under the Business Source License 1.1.
   See LICENSE in the repository root. *)

module ProofState

(* Trusted proof-state kernel. Neural scores are not part of this module. *)

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
