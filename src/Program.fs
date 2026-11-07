// Copyright (c) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at https://mariadb.com/bsl11
// or in the LICENSE file in the root of this repository.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

open System
open System.IO
open System.Text.RegularExpressions
open System.Security.Cryptography
open System.Text
open System.Numerics
open System.Globalization

module Text =
    let private culture = CultureInfo.InvariantCulture
    let fixed6 (value: float) = value.ToString("F6", culture)
    let general (value: float) = value.ToString("G17", culture)

module Tensor =
    type Matrix = float[,]
    let rows (a: Matrix) = a.GetLength 0
    let cols (a: Matrix) = a.GetLength 1
    let matmul (a: Matrix) (b: Matrix) : Matrix =
        if cols a <> rows b then invalidArg "b" "incompatible matrix dimensions"
        Array2D.init (rows a) (cols b) (fun i j ->
            let mutable s = 0.0
            for k in 0 .. cols a - 1 do s <- s + a[i,k] * b[k,j]
            s)
    let transpose (a: Matrix) : Matrix = Array2D.init (cols a) (rows a) (fun i j -> a[j,i])
    let map (f: float -> float) (a: Matrix) : Matrix = Array2D.init (rows a) (cols a) (fun i j -> f a[i,j])
    let add (a: Matrix) (b: Matrix) : Matrix =
        if rows a <> rows b || cols a <> cols b then invalidArg "b" "matrix shape mismatch"
        Array2D.init (rows a) (cols a) (fun i j -> a[i,j] + b[i,j])
    let softmaxRowsCausal (a: Matrix) : Matrix =
        if rows a <> cols a then invalidArg "a" "causal attention expects square scores"
        let n = rows a
        let output = Array2D.zeroCreate<float> n n
        for i in 0 .. n-1 do
            let mutable mx = Double.NegativeInfinity
            for j in 0 .. i do mx <- max mx a[i,j]
            let mutable denom = 0.0
            for j in 0 .. i do
                let e = exp (a[i,j] - mx)
                output[i,j] <- e
                denom <- denom + e
            for j in 0 .. i do output[i,j] <- output[i,j] / denom
        output

module Symbolic =
    // Supported roff subset: .ds NAME PATTERN => REPLACEMENT
    // Regex rewrites are deliberately bounded; they are not proof rules.
    let maxPasses = 64
    let maxOutputChars = 8192
    [<NoComparison>]
    type Rule = { Name: string; Pattern: Regex; Replacement: string }
    let load (path: string) =
        File.ReadAllLines path
        |> Array.choose (fun line ->
            let m = Regex.Match(line, @"^\.ds\s+(\w+)\s+(.+?)\s+=>\s+(.*)$")
            if not m.Success then None else
            Some { Name = m.Groups[1].Value
                   Pattern = Regex(m.Groups[2].Value, RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds 100.0)
                   Replacement = m.Groups[3].Value })
    let rewrite (rules: Rule array) (input: string) =
        if input.Length > maxOutputChars then
            input, 0, true
        else
            let mutable text = input
            let mutable steps = 0
            let mutable changed = true
            let mutable limited = false
            while changed && steps < maxPasses && not limited do
                changed <- false
                let mutable index = 0
                while index < rules.Length && not limited do
                    let rule = rules[index]
                    let next =
                        try rule.Pattern.Replace(text, rule.Replacement)
                        with :? RegexMatchTimeoutException ->
                            failwithf "rewrite rule %s timed out" rule.Name
                    if next <> text then
                        if next.Length > maxOutputChars then limited <- true
                        else
                            text <- next
                            changed <- true
                    index <- index + 1
                if not limited then steps <- steps + 1
            if changed && steps >= maxPasses then limited <- true
            text, steps, limited

module Model =
    open Tensor
    let defaultVocab = 128
    let defaultDim = 16
    let defaultHidden = 32
    let defaultSeed = 42
    type Weights = { Emb: Matrix; Wq: Matrix; Wk: Matrix; Wv: Matrix; Wo: Matrix; W1: Matrix; W2: Matrix; Head: Matrix }
    let init vocab dim hidden seed =
        let rng = Random(seed)
        let make r c = Array2D.init r c (fun _ _ -> (rng.NextDouble() * 2.0 - 1.0) * 0.15)
        { Emb = make vocab dim; Wq = make dim dim; Wk = make dim dim
          Wv = make dim dim; Wo = make dim dim; W1 = make dim hidden
          W2 = make hidden dim; Head = make dim vocab }
    let defaultWeights () = init defaultVocab defaultDim defaultHidden defaultSeed
    let encode (text: string) =
        if text.Length = 0 then invalidArg "text" "empty input"
        text.ToCharArray() |> Array.map (fun c -> int c % defaultVocab)
    let forward (w: Weights) (tokens: int array) =
        if tokens.Length = 0 then invalidArg "tokens" "empty input"
        let dim = cols w.Emb
        let x = Array2D.init tokens.Length dim (fun i j ->
            if tokens[i] < 0 || tokens[i] >= rows w.Emb then invalidArg "tokens" "out of vocabulary"
            w.Emb[tokens[i],j])
        let q = matmul x w.Wq
        let k = matmul x w.Wk
        let v = matmul x w.Wv
        let scores = matmul q (transpose k) |> map (fun z -> z / sqrt (float dim))
        let attn = matmul (softmaxRowsCausal scores) v |> fun z -> matmul z w.Wo
        let h = add x attn
        let ff = matmul h w.W1 |> map (fun z -> max 0.0 z) |> fun z -> matmul z w.W2
        matmul (add h ff) w.Head
    let nextToken (logits: Matrix) =
        let i = rows logits - 1
        [| 0 .. cols logits - 1 |] |> Array.maxBy (fun j -> logits[i,j])

module Checkpoint =
    open Tensor
    open Model
    let private magic = 0x424C4D31 // BLM1 as a little-endian int32
    let private version = 2
    let private maxFileBytes = 50L * 1024L * 1024L // 50 MiB hard cap
    let private matrices (w: Weights) = [|w.Emb; w.Wq; w.Wk; w.Wv; w.Wo; w.W1; w.W2; w.Head|]
    let private bytesEqual (a: byte[]) (b: byte[]) =
        a.Length = b.Length && Array.forall2 (fun x y -> x = y) a b
    let private validate (w: Weights) =
        let v, d = rows w.Emb, cols w.Emb
        let h = cols w.W1
        if v < 1 || d < 1 || h < 1 then failwith "invalid dimensions"
        let expect (a: Matrix) r c = if rows a <> r || cols a <> c then failwith "checkpoint shape mismatch"
        for a in [|w.Wq;w.Wk;w.Wv;w.Wo|] do expect a d d
        expect w.W1 d h; expect w.W2 h d; expect w.Head d v
        for a in matrices w do
            for i in 0 .. rows a - 1 do
                for j in 0 .. cols a - 1 do
                    let value = a[i,j]
                    if Double.IsNaN value || Double.IsInfinity value then failwith "nonfinite weight"
    let private writeMatrix (bw: BinaryWriter) (a: Matrix) =
        bw.Write(rows a); bw.Write(cols a)
        for i in 0..rows a-1 do for j in 0..cols a-1 do bw.Write(a[i,j])
    let save path (w: Weights) =
        validate w
        let tmp = path + ".tmp"
        try
            use ms = new MemoryStream()
            use bw = new BinaryWriter(ms, Encoding.UTF8, true)
            bw.Write(magic); bw.Write(version); bw.Write(8)
            for a in matrices w do writeMatrix bw a
            bw.Flush()
            let payload = ms.ToArray()
            if int64 payload.Length > maxFileBytes then failwith "checkpoint exceeds size cap"
            use sha = SHA256.Create()
            let hash = sha.ComputeHash(payload)
            do
                use fs = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None)
                fs.Write(payload, 0, payload.Length)
                fs.Write(hash, 0, hash.Length)
                fs.Flush(true)
            File.Move(tmp, path, true)
        finally
            if File.Exists(tmp) then File.Delete(tmp)
    let load path =
        let info = FileInfo(path)
        if info.Length > maxFileBytes + 32L then failwith "checkpoint exceeds size cap"
        use fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read)
        use br = new BinaryReader(fs)
        let payloadLen = int (fs.Length - 32L)
        if payloadLen < 16 then failwith "checkpoint too small"
        let payload = br.ReadBytes(payloadLen)
        let storedHash = br.ReadBytes(32)
        if br.BaseStream.Position <> br.BaseStream.Length then failwith "checkpoint has trailing bytes"
        use sha = SHA256.Create()
        let computed = sha.ComputeHash(payload)
        if not (bytesEqual storedHash computed) then failwith "checkpoint digest mismatch"
        use ms = new MemoryStream(payload)
        use pr = new BinaryReader(ms)
        if pr.ReadInt32() <> magic || pr.ReadInt32() <> version || pr.ReadInt32() <> 8 then failwith "invalid checkpoint header"
        let readMatrix () =
            let r, c = pr.ReadInt32(), pr.ReadInt32()
            if r < 1 || c < 1 || int64 r * int64 c > 10000000L then failwith "invalid checkpoint matrix dimensions"
            Array2D.init r c (fun _ _ -> pr.ReadDouble())
        let a = Array.init 8 (fun _ -> readMatrix())
        if ms.Position <> ms.Length then failwith "checkpoint payload has trailing data"
        let w = { Emb=a[0]; Wq=a[1]; Wk=a[2]; Wv=a[3]; Wo=a[4]; W1=a[5]; W2=a[6]; Head=a[7] }
        validate w
        w

module Reasoning =
    type Expr = Const of bigint | Var of string | Add of Expr * Expr | Mul of Expr * Expr
    type Rule = AddZeroL | AddZeroR | MulOneL | MulOneR | MulZeroL | MulZeroR | FoldAdd | FoldMul
    type Step = { Rule: Rule; Before: Expr; After: Expr }
    let rec show = function
        | Const n -> string n | Var s -> s
        | Add(a,b) -> "(" + show a + "+" + show b + ")"
        | Mul(a,b) -> "(" + show a + "*" + show b + ")"
    let local = function
        | Add(Const a,x) when a = 0I -> Some(AddZeroL,x)
        | Add(x,Const a) when a = 0I -> Some(AddZeroR,x)
        | Mul(Const a,x) when a = 1I -> Some(MulOneL,x)
        | Mul(x,Const a) when a = 1I -> Some(MulOneR,x)
        | Mul(Const a,_) when a = 0I -> Some(MulZeroL,Const 0I)
        | Mul(_,Const a) when a = 0I -> Some(MulZeroR,Const 0I)
        | Add(Const a,Const b) -> Some(FoldAdd,Const(a+b))
        | Mul(Const a,Const b) -> Some(FoldMul,Const(a*b))
        | _ -> None
    let rec successors expr =
        seq {
            match local expr with Some(r,v) -> yield r,v | None -> ()
            match expr with
            | Add(a,b) ->
                for r,v in successors a do yield r, Add(v,b)
                for r,v in successors b do yield r, Add(a,v)
            | Mul(a,b) ->
                for r,v in successors a do yield r, Mul(v,b)
                for r,v in successors b do yield r, Mul(a,v)
            | _ -> ()
        }
    // Bounded BFS, deterministic rule order; returns a shortest trace to a normal form.
    let search maxNodes input =
        if maxNodes < 1 then invalidArg "maxNodes" "must be positive"
        let queue = Collections.Generic.Queue<Expr * Step list>()
        let visited = Collections.Generic.HashSet<Expr>()
        queue.Enqueue((input, [])); visited.Add(input) |> ignore
        let mutable answer = None
        while queue.Count > 0 && answer.IsNone do
            let expr, trace = queue.Dequeue()
            let next = successors expr |> Seq.distinctBy snd |> Seq.toArray
            if next.Length = 0 then answer <- Some(expr, List.rev trace)
            else
                for rule, result in next do
                    if visited.Count < maxNodes && visited.Add(result) then
                        queue.Enqueue((result, {Rule=rule;Before=expr;After=result}::trace))
        answer
    let rec evaluate env = function
        | Const n -> n | Var v -> env v
        | Add(a,b) -> evaluate env a + evaluate env b
        | Mul(a,b) -> evaluate env a * evaluate env b
    // A certificate is accepted only when each claimed step follows a built-in rule.
    let verifyTrace (input: Expr) (target: Expr) (steps: Step list) =
        let mutable state = input
        let mutable valid = true
        for step in steps do
            if step.Before <> state ||
               not (successors state |> Seq.exists (fun (rule, result) -> rule = step.Rule && result = step.After)) then
                valid <- false
            else state <- step.After
        valid && state = target
    // Recursive-descent parser: identifiers, arbitrary precision nonnegative integers, parentheses, + and *.
    let parse (source:string) =
        let tokens = Regex.Matches(source, @"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[()+*]") |> Seq.cast<Match> |> Seq.map (fun m -> m.Value) |> Seq.toArray
        if String.Concat(tokens) <> Regex.Replace(source, @"\s+", "") then failwith "invalid expression token"
        let mutable at = 0
        let peek () = if at < tokens.Length then tokens[at] else ""
        let eat () = let t = peek() in at <- at+1; t
        let rec atom () =
            let t = eat()
            if t = "(" then
                let e = sum()
                if eat() <> ")" then failwith "missing closing parenthesis"
                e
            else
                match BigInteger.TryParse(t, NumberStyles.Integer, CultureInfo.InvariantCulture) with
                | true,n -> Const n
                | _ when Regex.IsMatch(t, @"^[A-Za-z_]") -> Var t
                | _ -> failwith "expected atom"
        and product () =
            let mutable x = atom()
            while peek() = "*" do eat() |> ignore; x <- Mul(x,atom())
            x
        and sum () =
            let mutable x = product()
            while peek() = "+" do eat() |> ignore; x <- Add(x,product())
            x
        let result = sum()
        if at <> tokens.Length then failwith "unexpected trailing token"
        result

module Guided =
    open Model
    open Reasoning
    open System.Collections.Generic
    // Feature vector from a proof state. Scores order candidates only.
    // The trusted checker never consults these scores.
    let exprSize (e: Expr) =
        let rec go = function
            | Const _ | Var _ -> 1
            | Add(a,b) | Mul(a,b) -> 1 + go a + go b
        go e
    let features (e: Expr) : float[] =
        let n = exprSize e
        let rec countVars = function
            | Const _ -> 0 | Var _ -> 1
            | Add(a,b) | Mul(a,b) -> countVars a + countVars b
        let rec countOps = function
            | Const _ | Var _ -> 0
            | Add(a,b) -> 1 + countOps a + countOps b
            | Mul(a,b) -> 1 + countOps a + countOps b
        [| float n; float (countVars e); float (countOps e); float (n % 8) |]
    // Deterministic score from the default multi-layer network. Ordering only.
    let score (w: Neural.Stack.Net) (e: Expr) : float =
        let feats = features e
        let tokens = feats |> Array.map (fun f -> int (abs f) % w.Config.Vocab)
        if tokens.Length = 0 then 0.0 else
        let logits = Neural.Stack.forward w tokens
        logits[tokens.Length - 1, 0]
    let private children expr =
        successors expr |> Seq.distinctBy snd |> Seq.toArray
    let private rank (w: Neural.Stack.Net) (left: Expr * Step list) (right: Expr * Step list) =
        let e1, _ = left
        let e2, _ = right
        let compared = compare (score w e2) (score w e1)
        if compared <> 0 then compared else compare (show e1) (show e2)
    // Beam search. Already-generated nodes are examined even when the node
    // budget was consumed by generating them, so a normal form at the budget
    // boundary is not dropped. Equal scores break ties by printed form.
    let beamSearch (w: Neural.Stack.Net) (beamWidth: int) (maxNodes: int) (input: Expr) =
        if beamWidth < 1 || maxNodes < 1 then invalidArg "limits" "must be positive"
        let mutable beam = [| input, ([]: Step list) |]
        let visited = HashSet<Expr>()
        visited.Add(input) |> ignore
        let mutable answer = None
        let mutable nodes = 1
        while beam.Length > 0 && answer.IsNone do
            let nextBeam = ResizeArray<Expr * Step list>()
            for expr, trace in beam do
                if answer.IsNone then
                    let next = children expr
                    if next.Length = 0 then answer <- Some(expr, List.rev trace)
                    else
                        for rule, result in next do
                            if nodes < maxNodes && visited.Add(result) then
                                nodes <- nodes + 1
                                nextBeam.Add((result, {Rule=rule;Before=expr;After=result}::trace))
            if answer.IsSome || nextBeam.Count = 0 then beam <- [||]
            else
                let ranked = nextBeam.ToArray()
                Array.sortInPlaceWith (rank w) ranked
                beam <- Array.truncate beamWidth ranked
        answer, nodes
    // Best-first search over the whole frontier, ordered by the same score.
    // Duplicate expressions are hashed once. maxNodes is the enqueue budget.
    let bestFirst (w: Neural.Stack.Net) (maxNodes: int) (input: Expr) =
        if maxNodes < 1 then invalidArg "maxNodes" "must be positive"
        let queue = PriorityQueue<Expr * Step list, float * string>()
        let visited = HashSet<Expr>()
        visited.Add(input) |> ignore
        queue.Enqueue((input, []), (-(score w input), show input))
        let mutable nodes = 1
        let mutable answer = None
        while queue.Count > 0 && answer.IsNone do
            let expr, trace = queue.Dequeue()
            let next = children expr
            if next.Length = 0 then answer <- Some(expr, List.rev trace)
            else
                for rule, result in next do
                    if nodes < maxNodes && visited.Add(result) then
                        nodes <- nodes + 1
                        let step = {Rule=rule;Before=expr;After=result}
                        queue.Enqueue((result, step::trace), (-(score w result), show result))
        answer, nodes
    type Strategy = Beam | BestFirst
    let parseStrategy = function
        | "beam" -> Beam
        | "best" -> BestFirst
        | other -> failwithf "unknown strategy '%s' (expected beam or best)" other
    let solve strategy w beamWidth maxNodes input =
        match strategy with
        | Beam -> beamSearch w beamWidth maxNodes input
        | BestFirst -> bestFirst w maxNodes input
    let ruleIndex = function
        | AddZeroL -> 0 | AddZeroR -> 1 | MulOneL -> 2 | MulOneR -> 3
        | MulZeroL -> 4 | MulZeroR -> 5 | FoldAdd -> 6 | FoldMul -> 7
    let trainingCorpus () =
        [ "(a+0)*(b*1)"; "x+0"; "0+y"; "z*1"; "0*w" ] |> List.map parse
    // Positives are applicable successor rules. One negative is the smallest
    // rule index that does not occur. This does not update the transformer.
    let generateExamples (exprs: Expr list) =
        [ for e in exprs do
            let steps = successors e |> Seq.toList
            for rule, _ in steps do
                yield features e, ruleIndex rule, true
            let used = steps |> List.map (fun (rule, _) -> ruleIndex rule) |> Set.ofList
            match List.tryFind (fun index -> not (Set.contains index used)) [ 0 .. 7 ] with
            | Some index -> yield features e, index, false
            | None -> () ]
    // Mean squared error of a linear proxy: sum(features) versus the rule index.
    // Invalid rows target -1. This is a measured proxy, not a training step.
    let measureRankingLoss (examples: (float[] * int * bool) list) =
        let mutable loss = 0.0
        let mutable n = 0
        for feats, ruleIdx, valid in examples do
            let pred = feats |> Array.sum
            let target = if valid then float ruleIdx else -1.0
            let err = pred - target
            loss <- loss + err * err
            n <- n + 1
        if n = 0 then 0.0 else loss / float n

module Bench =
    open Model
    type Run = {
        Iterations: int
        SeqLen: int
        Vocab: int
        Dim: int
        Hidden: int
        TotalMs: float
        PerMs: float
        Checksum: float
    }
    let measure () =
        let w = defaultWeights()
        let tokens = [|65;66;67;68;69;70;71;72|]
        for _ in 1..20 do Model.forward w tokens |> ignore
        let runCount = 200
        let sw = Diagnostics.Stopwatch.StartNew()
        let mutable digest = 0.0
        for _ in 1..runCount do
            let logits = Model.forward w tokens
            digest <- digest + logits[7,0]
        sw.Stop()
        let total = sw.Elapsed.TotalMilliseconds
        { Iterations = runCount; SeqLen = tokens.Length; Vocab = defaultVocab; Dim = defaultDim
          Hidden = defaultHidden; TotalMs = total; PerMs = total / float runCount; Checksum = digest }
    let format (run: Run) =
        sprintf "benchmark,os=%s,runtime=%s,iterations=%d,seq=%d,vocab=%d,dim=%d,hidden=%d,total_ms=%s,ms_per_forward=%s,checksum=%s"
            Environment.OSVersion.VersionString
            Runtime.InteropServices.RuntimeInformation.FrameworkDescription
            run.Iterations run.SeqLen run.Vocab run.Dim run.Hidden
            (Text.fixed6 run.TotalMs) (Text.fixed6 run.PerMs) (Text.general run.Checksum)
    let private median (values: float[]) =
        let sorted = Array.sort values
        let n = sorted.Length
        if n % 2 = 1 then sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    let private cpuName () =
        try
            File.ReadAllLines("/proc/cpuinfo")
            |> Array.tryFind (fun line -> line.StartsWith("model name"))
            |> Option.map (fun line -> line.Split(':', 2).[1].Trim())
            |> Option.defaultValue "unknown"
        with _ -> "unknown"
    let reportSuite () =
        let cfg = Neural.Stack.defaultConfig
        printfn "bench_env,os=%s,cpu=%s,logical_cpus=%d,runtime=%s,seed=%d,vocab=%d,dim=%d,heads=%d,layers=%d,hidden=%d"
            Environment.OSVersion.VersionString
            (cpuName())
            Environment.ProcessorCount
            Runtime.InteropServices.RuntimeInformation.FrameworkDescription
            Neural.Stack.defaultSeed cfg.Vocab cfg.Dim cfg.Heads cfg.Layers cfg.Hidden
        let repeats = 5
        let tokens = [|65; 66; 67; 68; 69; 70; 71; 72|]
        let inference () =
            let net = Neural.Stack.initDefault()
            for _ in 1 .. 10 do Neural.Stack.forward net tokens |> ignore
            let sw = Diagnostics.Stopwatch.StartNew()
            let mutable digest = 0.0
            for _ in 1 .. 100 do
                let logits = Neural.Stack.forward net tokens
                digest <- digest + logits[7, 0]
            sw.Stop()
            sw.Elapsed.TotalMilliseconds, digest
        let search () =
            let expr = Reasoning.parse "(a+0)*(b*1)"
            let sw = Diagnostics.Stopwatch.StartNew()
            let mutable steps = 0
            for _ in 1 .. 200 do
                match Reasoning.search 1000 expr with
                | Some(_, trace) -> steps <- steps + trace.Length
                | None -> ()
            sw.Stop()
            sw.Elapsed.TotalMilliseconds, float steps
        let training () =
            let corpus = Array.init 240 (fun i -> int ("abc".[i % 3]))
            let net = Neural.Stack.initDefault()
            let sw = Diagnostics.Stopwatch.StartNew()
            let losses = Neural.Stack.train net corpus 20 12 0.2
            sw.Stop()
            sw.Elapsed.TotalMilliseconds, losses[losses.Length - 1]
        let report name samples =
            let times = samples |> Array.map fst
            let marks = samples |> Array.map snd
            printfn "bench_%s,repeats=%d,min_ms=%s,median_ms=%s,max_ms=%s,checksum=%s"
                name samples.Length (Text.fixed6 (Array.min times)) (Text.fixed6 (median times)) (Text.fixed6 (Array.max times)) (Text.general marks[0])
            for i in 0 .. samples.Length - 1 do
                printfn "bench_%s_run,index=%d,ms=%s,checksum=%s" name (i + 1) (Text.fixed6 times[i]) (Text.general marks[i])
        report "inference" (Array.init repeats (fun _ -> inference()))
        report "search" (Array.init repeats (fun _ -> search()))
        report "train" (Array.init repeats (fun _ -> training()))

module Args =
    type ReasonOpts = { Expression: string; Nodes: int }
    type GuidedOpts = { Expression: string; Beam: int; Nodes: int; Strategy: string }
    let usageReason = "usage: reason <expression> [--nodes N]"
    let usageGuided = "usage: guided <expression> [--strategy beam|best] [--beam N] [--nodes N]"
    let usageSave = "usage: save <checkpoint.blm>"
    let usageLoad = "usage: load <checkpoint.blm> <prompt>"
    let usageRewrite = "usage: rewrite <rules.roff> <text>"
    let private parseInt flag (text: string) =
        let mutable value = 0
        if Int32.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, &value) then value
        else failwithf "%s expects an integer, got '%s'" flag text
    let parseReason (argv: string array) =
        if argv.Length < 2 then failwith usageReason
        let parts = ResizeArray<string>()
        let mutable nodes = 10000
        let mutable i = 1
        while i < argv.Length do
            match argv[i] with
            | "--nodes" ->
                if i + 1 >= argv.Length then failwith usageReason
                nodes <- parseInt "--nodes" argv[i+1]
                i <- i + 2
            | "--beam" | "--strategy" ->
                failwith "reason does not accept guided options; see: help"
            | token ->
                parts.Add(token)
                i <- i + 1
        if parts.Count = 0 then failwith usageReason
        { Expression = String.Join(" ", parts); Nodes = nodes }
    let parseGuided (argv: string array) =
        if argv.Length < 2 then failwith usageGuided
        let parts = ResizeArray<string>()
        let mutable beam = 4
        let mutable nodes = 1000
        let mutable strategy = "beam"
        let mutable i = 1
        while i < argv.Length do
            match argv[i] with
            | "--nodes" ->
                if i + 1 >= argv.Length then failwith usageGuided
                nodes <- parseInt "--nodes" argv[i+1]
                i <- i + 2
            | "--beam" ->
                if i + 1 >= argv.Length then failwith usageGuided
                beam <- parseInt "--beam" argv[i+1]
                i <- i + 2
            | "--strategy" ->
                if i + 1 >= argv.Length then failwith usageGuided
                strategy <- argv[i+1]
                i <- i + 2
            | token ->
                parts.Add(token)
                i <- i + 1
        if parts.Count = 0 then failwith usageGuided
        { Expression = String.Join(" ", parts); Beam = beam; Nodes = nodes; Strategy = strategy }

module Theorems =
    type Status = Proved | Disproved | Unknown
    type Result = { Name: string; Status: Status; Steps: int }
    let load (path: string) =
        File.ReadAllLines path
        |> Array.choose (fun line ->
            let m = Regex.Match(line, @"^\.thm\s+(\w+)\s+(.+?)\s+=\s+(.+)$")
            if m.Success then Some (m.Groups[1].Value, m.Groups[2].Value.Trim(), m.Groups[3].Value.Trim()) else None)
    let check maxNodes (name, left, right) =
        let lhs = Reasoning.parse left
        let rhs = Reasoning.parse right
        match Reasoning.search maxNodes lhs, Reasoning.search maxNodes rhs with
        | Some(normalLeft, traceLeft), Some(normalRight, traceRight) ->
            if not (Reasoning.verifyTrace lhs normalLeft traceLeft && Reasoning.verifyTrace rhs normalRight traceRight) then
                failwith "invalid theorem certificate"
            if Reasoning.show normalLeft = Reasoning.show normalRight then
                { Name = name; Status = Proved; Steps = traceLeft.Length + traceRight.Length }
            else
                { Name = name; Status = Disproved; Steps = traceLeft.Length + traceRight.Length }
        | _ -> { Name = name; Status = Unknown; Steps = 0 }

module Checks =
    open Tensor
    let private sameArray (left: 'a[]) (right: 'a[]) =
        left.Length = right.Length && Array.forall2 (fun x y -> x = y) left right
    let private expectMessage (substr: string) (action: unit -> unit) =
        let mutable message = ""
        let mutable threw = false
        try action() with ex ->
            threw <- true
            message <- ex.Message
        if not threw then failwithf "expected an error containing '%s'" substr
        if message.IndexOf(substr, StringComparison.Ordinal) < 0 then
            failwithf "expected error containing '%s' but got '%s'" substr message
    let private withTemp (suffix: string) (action: string -> unit) =
        let path = Path.Combine(Path.GetTempPath(), "bootleglm-" + Guid.NewGuid().ToString("N") + suffix)
        try action path
        finally if File.Exists(path) then File.Delete(path)
    let private repoRoot () =
        let rec walk (dir: string) =
            if File.Exists(Path.Combine(dir, "BootLegLM.fsproj")) &&
               File.Exists(Path.Combine(dir, "examples", "rules.roff")) then Some dir
            else
                match Directory.GetParent(dir) with
                | null -> None
                | parent -> walk parent.FullName
        match [| Directory.GetCurrentDirectory(); AppContext.BaseDirectory |] |> Array.tryPick walk with
        | Some dir -> dir
        | None -> failwith "could not locate examples/rules.roff"
    let rec private freeVars expr =
        match expr with
        | Reasoning.Const _ -> Set.empty
        | Reasoning.Var name -> Set.singleton name
        | Reasoning.Add(a, b) | Reasoning.Mul(a, b) -> Set.union (freeVars a) (freeVars b)
    let private assertEquivalent left right =
        let names = Set.union (freeVars left) (freeVars right) |> Set.toList
        let rec assign pending bound =
            match pending with
            | [] ->
                let env name =
                    match List.tryFind (fun (n, _) -> n = name) bound with
                    | Some (_, value) -> value
                    | None -> failwith "unbound variable"
                if Reasoning.evaluate env left <> Reasoning.evaluate env right then
                    failwithf "rewrite changed the value of %s vs %s" (Reasoning.show left) (Reasoning.show right)
            | name :: rest ->
                for value in -2 .. 2 do assign rest ((name, bigint value) :: bound)
        assign names []
    let private showTrace (steps: Reasoning.Step list) =
        steps
        |> List.map (fun step -> sprintf "%A %s => %s" step.Rule (Reasoning.show step.Before) (Reasoning.show step.After))
        |> String.concat " | "
    let private tensorAndModel () =
        let assertClose a b = if abs (a-b) > 1e-10 then failwithf "expected %g got %g" b a
        let a = array2D [[1.0;2.0];[3.0;4.0]]
        let b = array2D [[5.0;6.0];[7.0;8.0]]
        let c = matmul a b
        assertClose c[0,0] 19.0
        assertClose c[1,1] 50.0
        expectMessage "incompatible" (fun () -> matmul (array2D [[1.0]]) b |> ignore)
        expectMessage "shape mismatch" (fun () -> add a (array2D [[1.0]]) |> ignore)
        let p = softmaxRowsCausal (array2D [[1.0;2.0];[3.0;4.0]])
        assertClose p[0,0] 1.0
        assertClose p[0,1] 0.0
        assertClose (p[1,0]+p[1,1]) 1.0
        expectMessage "square" (fun () -> softmaxRowsCausal (array2D [[1.0; 2.0; 3.0]]) |> ignore)
        if Model.nextToken (array2D [[0.0; 5.0; 1.0]; [1.0; -1.0; 3.0]]) <> 2 then failwith "next token argmax"
        let w = Model.defaultWeights()
        let again = Model.defaultWeights()
        if w.Emb[0,0] <> again.Emb[0,0] then failwith "default weights are not deterministic"
        expectMessage "empty input" (fun () -> Model.forward w [||] |> ignore)
        expectMessage "out of vocabulary" (fun () -> Model.forward w [|Model.defaultVocab|] |> ignore)
        if not (sameArray (Model.encode "HELLO") [|72;69;76;76;79|]) then failwith "HELLO encoding"
        expectMessage "empty input" (fun () -> Model.encode "" |> ignore)
        let wrapped = Model.encode (string (char 200))
        if wrapped.[0] <> 200 % Model.defaultVocab then failwith "token wrap"
        let result = Model.forward w [|65;66;67|]
        if rows result <> 3 || cols result <> 128 then failwith "incorrect output shape"
        let short = Model.forward w [|65;66|]
        for j in 0..127 do assertClose short[1,j] result[1,j]
        if Model.nextToken (Model.forward w (Model.encode "HELLO")) <> 42 then failwith "HELLO next token changed"
        let mutable context = Model.encode "BOOTLEG"
        let generated = ResizeArray<int>()
        for _ in 1..16 do
            let next = Model.forward w context |> Model.nextToken
            generated.Add(next)
            context <- Array.append context [|next|]
        let expected = [|53;33;16;74;19;41;46;55;124;10;124;10;124;10;124;10|]
        if not (sameArray (generated.ToArray()) expected) then
            failwithf "generation changed: %s" (generated |> Seq.map string |> String.concat " ")
    let private checkpoint () =
        let w = Model.defaultWeights()
        withTemp ".blm" (fun path ->
            Checkpoint.save path w
            let restored = Checkpoint.load path
            let before = Model.forward w [|65;66;67|]
            let after = Model.forward restored [|65;66;67|]
            for i in 0 .. 2 do
                for j in 0 .. 127 do
                    if before[i,j] <> after[i,j] then failwith "checkpoint roundtrip changed logits"
            Checkpoint.save path w
            let reloaded = Checkpoint.load path
            if reloaded.Emb[0,0] <> w.Emb[0,0] then failwith "overwrite save failed"
            withTemp ".blm" (fun other ->
                Checkpoint.save other w
                if not (sameArray (File.ReadAllBytes path) (File.ReadAllBytes other)) then failwith "checkpoint bytes are not deterministic"
                let bytes = File.ReadAllBytes path
                bytes[bytes.Length - 1] <- bytes[bytes.Length - 1] ^^^ 255uy
                File.WriteAllBytes(other, bytes)
                expectMessage "digest" (fun () -> Checkpoint.load other |> ignore)
                let payload = bytes.[0 .. bytes.Length - 33]
                payload[0] <- payload[0] ^^^ 255uy
                File.WriteAllBytes(other, Array.append payload (SHA256.HashData payload))
                expectMessage "invalid checkpoint header" (fun () -> Checkpoint.load other |> ignore)
                let trailing = Array.append (File.ReadAllBytes(path).[0 .. File.ReadAllBytes(path).Length - 33]) [|0uy;0uy;0uy;0uy|]
                File.WriteAllBytes(other, Array.append trailing (SHA256.HashData trailing))
                expectMessage "trailing data" (fun () -> Checkpoint.load other |> ignore))
            use stream = new FileStream(path, FileMode.Append)
            stream.WriteByte(0uy)
            stream.Dispose()
            expectMessage "digest" (fun () -> Checkpoint.load path |> ignore))
        withTemp ".blm" (fun path ->
            let nanHead = Tensor.map (fun _ -> Double.NaN) w.Head
            expectMessage "nonfinite" (fun () -> Checkpoint.save path { w with Head = nanHead })
            expectMessage "shape" (fun () -> Checkpoint.save path { w with W1 = Array2D.zeroCreate 3 3 })
            File.WriteAllBytes(path, [|0uy|])
            expectMessage "too small" (fun () -> Checkpoint.load path |> ignore)
            use ms = new MemoryStream()
            use bw = new BinaryWriter(ms)
            bw.Write(0x424C4D31)
            bw.Write(2)
            bw.Write(8)
            bw.Write(10000001)
            bw.Write(1)
            bw.Flush()
            let payload = ms.ToArray()
            File.WriteAllBytes(path, Array.append payload (SHA256.HashData payload))
            expectMessage "invalid checkpoint matrix dimensions" (fun () -> Checkpoint.load path |> ignore))
    let private reasoning () =
        let expr = Reasoning.parse "(a+0)*(b*1)"
        if Reasoning.parse (Reasoning.show expr) <> expr then failwith "show/parse roundtrip"
        if Reasoning.parse " ( a + 0 ) " <> Reasoning.parse "(a+0)" then failwith "whitespace parse"
        let env0 (_: string) = 0I
        if Reasoning.evaluate env0 (Reasoning.parse "1+2*3") <> 7I then failwith "precedence 1+2*3"
        if Reasoning.evaluate env0 (Reasoning.parse "2*3+4") <> 10I then failwith "precedence 2*3+4"
        expectMessage "atom" (fun () -> Reasoning.parse "" |> ignore)
        expectMessage "atom" (fun () -> Reasoning.parse "a+" |> ignore)
        expectMessage "parenthesis" (fun () -> Reasoning.parse "(a" |> ignore)
        expectMessage "trailing" (fun () -> Reasoning.parse "a)" |> ignore)
        expectMessage "token" (fun () -> Reasoning.parse "-1" |> ignore)
        let zero = Reasoning.parse "a+0"
        let good : Reasoning.Step = { Rule = Reasoning.AddZeroR; Before = zero; After = Reasoning.parse "a" }
        if not (Reasoning.verifyTrace zero (Reasoning.parse "a") [good]) then failwith "valid root step rejected"
        let bad = { good with Rule = Reasoning.AddZeroL }
        if Reasoning.verifyTrace zero (Reasoning.parse "a") [bad] then failwith "wrong rule accepted"
        if not (Reasoning.verifyTrace zero zero []) then failwith "empty certificate rejected for identical terms"
        if Reasoning.verifyTrace zero (Reasoning.parse "a") [] then failwith "empty certificate skipped a step"
        match Reasoning.search 100 expr with
        | Some(normal, trace) ->
            if Reasoning.show normal <> "(a*b)" then failwith "wrong normal form"
            assertEquivalent expr normal
            if trace.Length <> 2 then failwith "unexpected search length"
            if not (Reasoning.verifyTrace expr normal trace) then failwith "invalid rewrite certificate"
            let tampered = { trace.Head with After = Reasoning.Const 999I } :: trace.Tail
            if Reasoning.verifyTrace expr normal tampered then failwith "tampered proof accepted"
        | None -> failwith "search failed"
        let large = Reasoning.parse "99999999999999999999999999999+1"
        match Reasoning.search 20 large with
        | Some(result, proof) when Reasoning.show result = "100000000000000000000000000000" && Reasoning.verifyTrace large result proof -> ()
        | _ -> failwith "arbitrary precision rewrite failed"
        let product = Reasoning.parse "100000000000000000000*100000000000000000000"
        match Reasoning.search 10 product with
        | Some(Reasoning.Const n, proof) when n = 100000000000000000000I * 100000000000000000000I && Reasoning.verifyTrace product (Reasoning.Const n) proof -> ()
        | _ -> failwith "arbitrary precision product failed"
        match Reasoning.search 3 expr with
        | None -> ()
        | Some _ -> failwith "node budget 3 should exhaust"
        match Reasoning.search 4 expr with
        | Some(normal, trace) when Reasoning.show normal = "(a*b)" && Reasoning.verifyTrace expr normal trace -> ()
        | _ -> failwith "node budget 4 should find the normal form"
        expectMessage "positive" (fun () -> Reasoning.search 0 expr |> ignore)
        let parsed = Args.parseReason [| "reason"; "--nodes"; "4"; "(a+0)*(b*1)" |]
        if parsed.Nodes <> 4 || parsed.Expression <> "(a+0)*(b*1)" then failwith "reason flag parse"
        let swapped = Args.parseReason [| "reason"; "a+0"; "--nodes"; "12" |]
        if swapped.Nodes <> 12 || swapped.Expression <> "a+0" then failwith "reason flag order"
        expectMessage "integer" (fun () -> Args.parseReason [| "reason"; "a"; "--nodes"; "x" |] |> ignore)
        expectMessage "usage" (fun () -> Args.parseReason [| "reason" |] |> ignore)
        expectMessage "guided" (fun () -> Args.parseReason [| "reason"; "a"; "--beam"; "2" |] |> ignore)
        let negative = Args.parseReason [| "reason"; "a+0"; "--nodes"; "-1" |]
        expectMessage "positive" (fun () -> Reasoning.search negative.Nodes (Reasoning.parse negative.Expression) |> ignore)
    let private guided () =
        let w = Neural.Stack.initDefault()
        let expr = Reasoning.parse "(a+0)*(b*1)"
        let assertSolved label input (found, nodes) =
            match found with
            | Some(normal, trace) ->
                if not (Reasoning.verifyTrace input normal trace) then failwithf "%s certificate rejected (%d nodes)" label nodes
                assertEquivalent input normal
                normal, trace
            | None -> failwithf "%s exhausted at %d nodes" label nodes
        let beamNormal, beamTrace = assertSolved "beam" expr (Guided.beamSearch w 4 1000 expr)
        if Reasoning.show beamNormal <> "(a*b)" || beamTrace.Length <> 2 then failwith "beam normal form"
        match Guided.beamSearch w 4 1000 expr with
        | Some _, nodes when nodes = 4 -> ()
        | Some _, nodes -> failwithf "beam node count %d" nodes
        | None, _ -> failwith "beam missed the diamond"
        if beamTrace.[0].Rule <> Reasoning.MulOneR || beamTrace.[1].Rule <> Reasoning.AddZeroR then
            failwithf "beam trace changed: %s" (showTrace beamTrace)
        let repeat, repeatTrace = assertSolved "beam-repeat" expr (Guided.beamSearch w 4 1000 expr)
        if Reasoning.show repeat <> Reasoning.show beamNormal || showTrace repeatTrace <> showTrace beamTrace then
            failwith "beam search is not deterministic"
        match Guided.beamSearch w 4 4 expr with
        | Some(normal, trace), _ when Reasoning.show normal = "(a*b)" && Reasoning.verifyTrace expr normal trace -> ()
        | _ -> failwith "beam dropped the normal form at the node budget"
        match Guided.beamSearch w 4 3 expr with
        | None, _ -> ()
        | Some _, _ -> failwith "beam budget 3 should exhaust"
        let normalExpr = Reasoning.parse "a+b"
        match Guided.beamSearch w 1 1 normalExpr with
        | Some(normal, trace), _ when normal = normalExpr && trace.IsEmpty -> ()
        | _ -> failwith "beam missed an already-normal input at budget 1"
        let bestNormal, bestTrace = assertSolved "best" expr (Guided.bestFirst w 1000 expr)
        if Reasoning.show bestNormal <> "(a*b)" then failwith "best-first normal form"
        let bestAgain, bestAgainTrace = assertSolved "best-repeat" expr (Guided.bestFirst w 1000 expr)
        if showTrace bestAgainTrace <> showTrace bestTrace || Reasoning.show bestAgain <> Reasoning.show bestNormal then
            failwith "best-first search is not deterministic"
        match Guided.bestFirst w 4 expr with
        | Some(normal, trace), _ when Reasoning.show normal = "(a*b)" && Reasoning.verifyTrace expr normal trace -> ()
        | _ -> failwith "best-first missed the normal form at budget 4"
        match Guided.bestFirst w 3 expr with
        | None, _ -> ()
        | Some _, _ -> failwith "best-first budget 3 should exhaust"
        match Guided.bestFirst w 1 normalExpr with
        | Some(normal, trace), _ when normal = normalExpr && trace.IsEmpty -> ()
        | _ -> failwith "best-first missed an already-normal input"
        expectMessage "positive" (fun () -> Guided.beamSearch w 0 10 expr |> ignore)
        expectMessage "unknown strategy" (fun () -> Guided.parseStrategy "astar" |> ignore)
        let opts = Args.parseGuided [| "guided"; "x+0"; "--strategy"; "best"; "--nodes"; "10"; "--beam"; "2" |]
        if opts.Strategy <> "best" || opts.Nodes <> 10 || opts.Beam <> 2 || opts.Expression <> "x+0" then
            failwith "guided flag parse"
        expectMessage "usage" (fun () -> Args.parseGuided [| "guided"; "--nodes"; "3" |] |> ignore)
        let samples = [
            "(a+0)*(b*1)"; "x+0"; "0+y"; "z*1"; "1*z"; "0*w"; "w*0"; "0*0"; "1*0"; "0*1"
            "2+3"; "2*3"; "(1+0)*(0+1)"; "(a+0)+((b*1)*1)"; "0*(a+5)"; "(2+3)*(4+0)"
            "99999999999999999999999999999+1"; "((1+2)+3)*1"; "a+b"; "(a)"; "0" ]
        for sample in samples do
            let input = Reasoning.parse sample
            match Reasoning.search 10000 input with
            | None -> failwithf "BFS exhausted on %s" sample
            | Some(bfsNormal, bfsTrace) ->
                if not (Reasoning.verifyTrace input bfsNormal bfsTrace) then failwithf "BFS certificate failed for %s" sample
                assertEquivalent input bfsNormal
                let beamN, _ = assertSolved "beam-suite" input (Guided.beamSearch w 4 1000 input)
                let bestN, _ = assertSolved "best-suite" input (Guided.bestFirst w 1000 input)
                if Reasoning.show beamN <> Reasoning.show bfsNormal || Reasoning.show bestN <> Reasoning.show bfsNormal then
                    failwithf "strategies disagreed on %s: bfs %s beam %s best %s" sample (Reasoning.show bfsNormal) (Reasoning.show beamN) (Reasoning.show bestN)
        if beamTrace.Length > 0 then
            let tampered = { beamTrace.Head with After = Reasoning.Const 999I } :: beamTrace.Tail
            if Reasoning.verifyTrace expr beamNormal tampered then failwith "tampered guided certificate accepted"
    let private ranking () =
        if Guided.measureRankingLoss [] <> 0.0 then failwith "empty ranking loss"
        let hand = Guided.measureRankingLoss [ ([|1.0; 1.0|], 1, true); ([|1.0; 1.0|], 0, false) ]
        if hand <> 5.0 then failwithf "ranking proxy changed: %g" hand
        let corpus = Guided.trainingCorpus()
        let examples = Guided.generateExamples corpus
        if examples.Length <> 11 then failwithf "example count %d" examples.Length
        for expr in corpus do
            let applicable =
                Reasoning.successors expr
                |> Seq.map (fun (rule, _) -> Guided.ruleIndex rule)
                |> Set.ofSeq
            let rows = Guided.generateExamples [expr]
            let negatives = rows |> List.filter (fun (_, _, valid) -> not valid)
            if applicable.Count < 8 && negatives.Length <> 1 then failwith "expected one negative label"
            for _, index, valid in rows do
                if valid <> Set.contains index applicable then
                    failwithf "mislabeled rule %d on %s (valid=%b)" index (Reasoning.show expr) valid
        let loss = Guided.measureRankingLoss examples
        let expected = 1458.0 / 11.0
        if abs (loss - expected) > 1e-9 then failwithf "corpus ranking loss %.12f vs %.12f" loss expected
        if Text.fixed6 loss <> "132.545455" then failwithf "ranking format %s" (Text.fixed6 loss)
        if Text.fixed6 1.5 <> "1.500000" then failwith "invariant format"
    let private rewrite () =
        let root = repoRoot()
        let rules = Symbolic.load (Path.Combine(root, "examples", "rules.roff"))
        if rules.Length <> 2 then failwithf "expected 2 example rules, got %d" rules.Length
        let text, steps, limit = Symbolic.rewrite rules "x + 0"
        if text <> "x" || steps <> 2 || limit then failwithf "ZERO rewrite %s passes=%d limit=%b" text steps limit
        let one, _, oneLimit = Symbolic.rewrite rules "x * 1"
        if one <> "x" || oneLimit then failwith "ONE rewrite"
        let twice, _, twiceLimit = Symbolic.rewrite rules "y + 0 + 0"
        if twice <> "y" || twiceLimit then failwithf "repeated ZERO -> %s" twice
        let left, leftSteps, leftLimit = Symbolic.rewrite rules "0 + x"
        if left <> "0 + x" || leftSteps <> 1 || leftLimit then failwith "regex front-end simplified 0 + x"
        match Reasoning.search 10 (Reasoning.parse "0+x") with
        | Some(normal, _) when Reasoning.show normal = "x" -> ()
        | _ -> failwith "structural rewriter did not simplify 0+x"
        let unchanged, confirm, confirmLimit = Symbolic.rewrite [||] "abc"
        if unchanged <> "abc" || confirm <> 1 || confirmLimit then failwith "empty rule set"
        withTemp ".roff" (fun path ->
            File.WriteAllText(path, "# comment\n.\\\" note\n\n.ds ZERO \\b([A-Za-z]+)\\s*\\+\\s*0\\b => $1\n")
            let loaded = Symbolic.load path
            if loaded.Length <> 1 then failwith "comments were not ignored"
            let simplified, _, limited = Symbolic.rewrite loaded "x + 0"
            if simplified <> "x" || limited then failwith "temp rule did not fire"
            File.WriteAllText(path, ".ds BAD ( => x\n")
            expectMessage "pattern" (fun () -> Symbolic.load path |> ignore)
            File.WriteAllText(path, ".ds AB ab => ba\n.ds BA ba => ab\n")
            let oscillated, passes, hit = Symbolic.rewrite (Symbolic.load path) "ab"
            if oscillated <> "ab" || passes <> Symbolic.maxPasses || not hit then
                failwithf "pass cap failed: %s passes=%d limit=%b" oscillated passes hit
            File.WriteAllText(path, ".ds GROW a => aa\n")
            let grown, growPasses, growLimit = Symbolic.rewrite (Symbolic.load path) "a"
            if not growLimit || grown.Length <> Symbolic.maxOutputChars || growPasses <> 13 then
                failwithf "size cap failed: len=%d passes=%d limit=%b" grown.Length growPasses growLimit
            if grown |> Seq.exists (fun ch -> ch <> 'a') then failwith "size cap corrupted text"
            let huge = String('b', Symbolic.maxOutputChars + 1)
            let kept, keptPasses, keptLimit = Symbolic.rewrite (Symbolic.load path) huge
            if kept <> huge || keptPasses <> 0 || not keptLimit then failwith "oversized input was rewritten"
            File.WriteAllText(path, ".ds REDOS (a+)+$ => b\n")
            expectMessage "timed out" (fun () ->
                Symbolic.rewrite (Symbolic.load path) (String('a', 30) + "X") |> ignore))
    let private closeEnough a b = abs (a - b) <= 1e-8
    let private stackTrainAndTheorems () =
        let cfg = Neural.Stack.defaultConfig
        if Neural.Stack.headDim cfg <> 4 then failwith "head dimension"
        expectMessage "divisible" (fun () -> Neural.Stack.init { cfg with Heads = 3 } 1 |> ignore)
        let net = Neural.Stack.initDefault()
        let long = Neural.Stack.forward net [|10; 20; 30; 40|]
        let short = Neural.Stack.forward net [|10; 20|]
        let one = Neural.Stack.forward net [|10|]
        if long.GetLength 0 <> 4 || long.GetLength 1 <> cfg.Vocab then failwith "stack logit shape"
        if short.GetLength 0 <> 2 || one.GetLength 0 <> 1 then failwith "stack prefix shape"
        for j in 0 .. cfg.Vocab - 1 do
            if not (closeEnough long[0, j] short[0, j] && closeEnough long[0, j] one[0, j]) then failwith "position 0 saw the future"
            if not (closeEnough long[1, j] short[1, j]) then failwith "position 1 saw the future"
        let tiny : Neural.Stack.Config = { Vocab = 6; Dim = 4; Heads = 2; Layers = 1; Hidden = 6 }
        let probe = Neural.Stack.init tiny 1
        let sample = [|1; 2; 3; 1; 2|]
        let _, grads = Neural.Stack.backward probe sample
        let checkGrad name analytic read write =
            let eps = 1e-4
            let original = read()
            write (original + eps)
            let up = Neural.Stack.loss probe sample
            write (original - eps)
            let down = Neural.Stack.loss probe sample
            write original
            let numeric = (up - down) / (2.0 * eps)
            if abs (numeric - analytic) > 1e-3 then failwithf "%s gradient %g vs numeric %g" name analytic numeric
        checkGrad "embedding" grads.Emb[1, 0] (fun () -> probe.Emb[1, 0]) (fun v -> probe.Emb[1, 0] <- v)
        checkGrad "query" grads.Layers[0].Wq[0, 1] (fun () -> probe.Layers[0].Wq[0, 1]) (fun v -> probe.Layers[0].Wq[0, 1] <- v)
        checkGrad "layernorm" grads.Layers[0].Ln1G[1] (fun () -> probe.Layers[0].Ln1G[1]) (fun v -> probe.Layers[0].Ln1G[1] <- v)
        checkGrad "head" grads.Head[1, 3] (fun () -> probe.Head[1, 3]) (fun v -> probe.Head[1, 3] <- v)
        checkGrad "ffn" grads.Layers[0].W1[0, 1] (fun () -> probe.Layers[0].W1[0, 1]) (fun v -> probe.Layers[0].W1[0, 1] <- v)
        let corpus = Neural.Stack.loadCorpus (Path.Combine(repoRoot(), "examples", "corpus.txt")) cfg.Vocab
        let trained = Neural.Stack.initDefault()
        let fresh = Neural.Stack.copy trained
        let losses = Neural.Stack.train trained corpus 40 12 0.2
        let again = Neural.Stack.initDefault()
        let lossesAgain = Neural.Stack.train again corpus 40 12 0.2
        if losses.Length <> 40 then failwith "training step count"
        if losses[0] <> lossesAgain[0] || losses[losses.Length - 1] <> lossesAgain[lossesAgain.Length - 1] then
            failwith "training is not reproducible"
        if not (losses[losses.Length - 1] < losses[0] - 0.5) then
            failwithf "loss did not improve: %g -> %g" losses[0] losses[losses.Length - 1]
        let mutable moved = 0.0
        for i in 0 .. cfg.Vocab - 1 do
            for j in 0 .. cfg.Dim - 1 do
                moved <- moved + abs (trained.Emb[i, j] - fresh.Emb[i, j])
        if moved = 0.0 then failwith "training did not update weights"
        if Double.IsNaN losses[losses.Length - 1] then failwith "final loss is not finite"
        withTemp ".blm" (fun path ->
            Neural.Stack.save path trained
            let restored = Neural.Stack.load path
            let a = Neural.Stack.forward trained [|65; 66; 67; 68|]
            let b = Neural.Stack.forward restored [|65; 66; 67; 68|]
            for i in 0 .. 3 do
                for j in 0 .. cfg.Vocab - 1 do
                    if a[i, j] <> b[i, j] then failwith "stack checkpoint changed logits"
            let bytes = File.ReadAllBytes path
            bytes[bytes.Length - 1] <- bytes[bytes.Length - 1] ^^^ 255uy
            File.WriteAllBytes(path, bytes)
            expectMessage "digest" (fun () -> Neural.Stack.load path |> ignore))
        let theorems = Theorems.load (Path.Combine(repoRoot(), "examples", "theorems.roff"))
        if theorems.Length < 4 then failwith "missing theorem declarations"
        for decl in theorems do
            let result = Theorems.check 10000 decl
            if result.Status <> Theorems.Proved then failwithf "theorem %s was %A" result.Name result.Status
        withTemp ".roff" (fun path ->
            File.WriteAllText(path, ".thm BAD a+b = a\n")
            let result = Theorems.check 100 (Theorems.load path).[0]
            if result.Status <> Theorems.Disproved then failwith "false theorem was not rejected")
    let private bench () =
        let first = Bench.measure()
        let second = Bench.measure()
        if first.Iterations <> 200 || first.SeqLen <> 8 || first.Vocab <> 128 || first.Dim <> 16 || first.Hidden <> 32 then
            failwith "benchmark metadata"
        if first.Checksum <> second.Checksum || Double.IsNaN first.Checksum || Double.IsInfinity first.Checksum then
            failwith "benchmark checksum is not a finite deterministic value"
        if abs (first.Checksum - (-8.4242520795687188)) > 1e-12 then
            failwithf "benchmark checksum changed: %s" (Text.general first.Checksum)
        let line = Bench.format first
        if not (line.Contains("checksum=") && line.Contains("ms_per_forward=")) then failwith "benchmark line"
    let run () =
        tensorAndModel()
        checkpoint()
        reasoning()
        guided()
        ranking()
        rewrite()
        stackTrainAndTheorems()
        bench()
        printfn "PASS: matmul, causal softmax, shapes, prefix invariance, checkpoint roundtrip/digest/shape/nonfinite, BFS budgets, certificates, big integers, guided beam and best-first, ranking labels, bounded regex rewrite, multi-head stack, gradients, training, v3 checkpoints, roff theorems"

module Cli =
    let printUsage () =
        printfn "BootLegLM"
        printfn "  test"
        printfn "  save <checkpoint.blm>"
        printfn "  load <checkpoint.blm> <prompt>"
        printfn "  train <corpus.txt> <checkpoint.blm> [--steps N] [--lr X] [--seq N] [--seed N]"
        printfn "  reason <expression> [--nodes N]"
        printfn "  guided <expression> [--strategy beam|best] [--beam N] [--nodes N]"
        printfn "  prove <theorems.roff> [--nodes N]"
        printfn "  traindata"
        printfn "  bench"
        printfn "  rewrite <rules.roff> <text>"
        printfn "  help"
        printfn "  <prompt>"
        printfn "Exit codes: 0 success, 1 error, 2 search or theorem budget exhausted."
        printfn "Default stack: seed %d, vocab %d, dim %d, heads %d, layers %d, hidden %d."
            Neural.Stack.defaultSeed Neural.Stack.defaultConfig.Vocab Neural.Stack.defaultConfig.Dim
            Neural.Stack.defaultConfig.Heads Neural.Stack.defaultConfig.Layers Neural.Stack.defaultConfig.Hidden
        printfn "save writes BLM1 version 3. load still accepts version 2 single-head checkpoints."
        printfn "train updates weights with SGD on next-character cross-entropy. Default steps=40, lr=0.2, seq=12."
        printfn "reason node budget defaults to 10000. guided defaults to beam width 4 and 1000 nodes."
        printfn "best-first ignores --beam and keeps the frontier until --nodes."
        printfn "prove checks .thm NAME lhs = rhs by certified normalization. .ds lines stay regex rewrites."
        printfn "traindata prints a ranking MSE proxy and does not update weights."
        printfn "Regex rewrite stops after %d passes or at %d characters." Symbolic.maxPasses Symbolic.maxOutputChars
        printfn "A bare prompt generates 16 tokens from the untrained default stack."
    let private peekVersion path =
        use fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read)
        use br = new BinaryReader(fs)
        let magic = br.ReadInt32()
        if magic <> 0x424C4D31 then failwith "invalid checkpoint header"
        if fs.Length < 8L then failwith "checkpoint too small"
        br.ReadInt32()
    let private parseDouble flag (text: string) =
        match Double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture) with
        | true, value -> value
        | _ -> failwithf "%s expects a number, got '%s'" flag text
    let private parseTrain (argv: string array) =
        if argv.Length < 3 then failwith "usage: train <corpus.txt> <checkpoint.blm> [--steps N] [--lr X] [--seq N] [--seed N]"
        let mutable steps = 40
        let mutable lr = 0.2
        let mutable seqLen = 12
        let mutable seed = Neural.Stack.defaultSeed
        let mutable i = 3
        while i < argv.Length do
            match argv[i] with
            | "--steps" ->
                if i + 1 >= argv.Length then failwith "usage: train ... [--steps N]"
                steps <- Int32.Parse(argv[i + 1], CultureInfo.InvariantCulture)
                i <- i + 2
            | "--lr" ->
                if i + 1 >= argv.Length then failwith "usage: train ... [--lr X]"
                lr <- parseDouble "--lr" argv[i + 1]
                i <- i + 2
            | "--seq" ->
                if i + 1 >= argv.Length then failwith "usage: train ... [--seq N]"
                seqLen <- Int32.Parse(argv[i + 1], CultureInfo.InvariantCulture)
                i <- i + 2
            | "--seed" ->
                if i + 1 >= argv.Length then failwith "usage: train ... [--seed N]"
                seed <- Int32.Parse(argv[i + 1], CultureInfo.InvariantCulture)
                i <- i + 2
            | _ -> failwith "usage: train <corpus.txt> <checkpoint.blm> [--steps N] [--lr X] [--seq N] [--seed N]"
        argv[1], argv[2], steps, lr, seqLen, seed
    let private parseProve (argv: string array) =
        if argv.Length < 2 then failwith "usage: prove <theorems.roff> [--nodes N]"
        let mutable nodes = 10000
        let mutable i = 2
        while i < argv.Length do
            match argv[i] with
            | "--nodes" ->
                if i + 1 >= argv.Length then failwith "usage: prove <theorems.roff> [--nodes N]"
                nodes <- Int32.Parse(argv[i + 1], CultureInfo.InvariantCulture)
                i <- i + 2
            | _ -> failwith "usage: prove <theorems.roff> [--nodes N]"
        argv[1], nodes
    let private printTrace (steps: Reasoning.Step list) =
        for step in steps do
            printfn "%A: %s -> %s" step.Rule (Reasoning.show step.Before) (Reasoning.show step.After)
    let run (argv: string array) =
        if argv.Length > 0 && (argv[0] = "help" || argv[0] = "--help" || argv[0] = "-h") then
            printUsage()
            0
        elif argv.Length > 0 && argv[0] = "test" then
            Checks.run()
            0
        elif argv.Length > 0 && argv[0] = "save" then
            if argv.Length <> 2 then failwith Args.usageSave
            Neural.Stack.save argv[1] (Neural.Stack.initDefault())
            printfn "Saved checkpoint: %s" argv[1]
            printfn "format=BLM1v3,layers=%d,heads=%d,dim=%d" Neural.Stack.defaultConfig.Layers Neural.Stack.defaultConfig.Heads Neural.Stack.defaultConfig.Dim
            0
        elif argv.Length > 0 && argv[0] = "load" then
            if argv.Length < 3 then failwith Args.usageLoad
            let tokens = Model.encode (String.Join(" ", argv[2..]))
            let version = peekVersion argv[1]
            let logits =
                if version = 2 then Model.forward (Checkpoint.load argv[1]) tokens
                elif version = Neural.Stack.version then Neural.Stack.forward (Neural.Stack.load argv[1]) tokens
                else failwith "unsupported checkpoint version"
            printfn "Next token: %d" (Model.nextToken logits)
            printfn "format=BLM1v%d" version
            0
        elif argv.Length > 0 && argv[0] = "train" then
            let corpus, output, steps, lr, seqLen, seed = parseTrain argv
            let cfg = Neural.Stack.defaultConfig
            let net = Neural.Stack.init cfg seed
            let tokens = Neural.Stack.loadCorpus corpus cfg.Vocab
            let losses = Neural.Stack.train net tokens steps seqLen lr
            Neural.Stack.save output net
            printfn "train,corpus=%s,steps=%d,seq=%d,lr=%s,seed=%d,layers=%d,heads=%d,dim=%d,hidden=%d,vocab=%d"
                corpus steps seqLen (Text.fixed6 lr) seed cfg.Layers cfg.Heads cfg.Dim cfg.Hidden cfg.Vocab
            for i in 0 .. losses.Length - 1 do
                printfn "loss_step=%d,loss=%s" i (Text.fixed6 losses[i])
            printfn "saved=%s,loss_initial=%s,loss_final=%s" output (Text.fixed6 losses[0]) (Text.fixed6 losses[losses.Length - 1])
            0
        elif argv.Length > 0 && argv[0] = "prove" then
            let path, nodes = parseProve argv
            let decls = Theorems.load path
            if decls.Length = 0 then failwith "no .thm declarations"
            let mutable unknown = 0
            let mutable disproved = 0
            for decl in decls do
                let result = Theorems.check nodes decl
                printfn "%s: %A (%d certificate steps)" result.Name result.Status result.Steps
                if result.Status = Theorems.Unknown then unknown <- unknown + 1
                if result.Status = Theorems.Disproved then disproved <- disproved + 1
            if disproved > 0 then 1
            elif unknown > 0 then 2
            else 0
        elif argv.Length > 0 && argv[0] = "reason" then
            let opts = Args.parseReason argv
            let expr = Reasoning.parse opts.Expression
            match Reasoning.search opts.Nodes expr with
            | None -> printfn "UNKNOWN: search exhausted"; 2
            | Some(normal, trace) ->
                if not (Reasoning.verifyTrace expr normal trace) then failwith "invalid search certificate"
                printTrace trace
                printfn "CERTIFICATE: verified %d steps" trace.Length
                printfn "RESULT: %s" (Reasoning.show normal)
                0
        elif argv.Length > 0 && argv[0] = "guided" then
            let opts = Args.parseGuided argv
            let expr = Reasoning.parse opts.Expression
            let w = Neural.Stack.initDefault()
            let strategy = Guided.parseStrategy opts.Strategy
            match Guided.solve strategy w opts.Beam opts.Nodes expr with
            | Some(normal, trace), nodes ->
                if not (Reasoning.verifyTrace expr normal trace) then failwith "invalid search certificate"
                printTrace trace
                printfn "CERTIFICATE: verified %d steps (nodes=%d)" trace.Length nodes
                printfn "RESULT: %s" (Reasoning.show normal)
                0
            | None, nodes ->
                printfn "UNKNOWN: search exhausted (nodes=%d)" nodes
                2
        elif argv.Length > 0 && argv[0] = "traindata" then
            let examples = Guided.generateExamples (Guided.trainingCorpus())
            let loss = Guided.measureRankingLoss examples
            printfn "examples=%d ranking_mse=%s" examples.Length (Text.fixed6 loss)
            0
        elif argv.Length > 0 && argv[0] = "bench" then
            Bench.reportSuite()
            let legacy = Bench.measure()
            printfn "bench_legacy_v2,%s" (Bench.format legacy)
            0
        elif argv.Length > 0 && argv[0] = "rewrite" then
            if argv.Length < 3 then failwith Args.usageRewrite
            let rules = Symbolic.load argv[1]
            let result, steps, limit = Symbolic.rewrite rules (String.Join(" ", argv[2..]))
            printfn "%s (passes=%d, limitReached=%b)" result steps limit
            0
        else
            let prompt = if argv.Length = 0 then "BOOTLEG" else String.Join(" ", argv)
            let net = Neural.Stack.initDefault()
            let mutable context = Model.encode prompt
            let generated = ResizeArray<char>()
            for _ in 1..16 do
                let next = Neural.Stack.forward net context |> Model.nextToken
                generated.Add(char next)
                context <- Array.append context [|next|]
            printfn "Input: %s" prompt
            printfn "Generated token IDs: %s" (generated |> Seq.map (int >> string) |> String.concat " ")
            printfn "WARNING: default stack is randomly initialized and has not been trained."
            0

[<EntryPoint>]
let main argv =
    try
        Cli.run argv
    with ex ->
        eprintfn "ERROR: %s" ex.Message
        1
