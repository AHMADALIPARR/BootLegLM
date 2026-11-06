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
open System.Text.Json
open System.Numerics
open System.Linq

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
        let mutable text = input
        let mutable steps = 0
        let mutable changed = true
        while changed && steps < 64 do
            changed <- false
            for rule in rules do
                let next = rule.Pattern.Replace(text, rule.Replacement)
                if next <> text then
                    text <- next
                    changed <- true
            steps <- steps + 1
        text, steps, changed

module Model =
    open Tensor
    type Weights = { Emb: Matrix; Wq: Matrix; Wk: Matrix; Wv: Matrix; Wo: Matrix; W1: Matrix; W2: Matrix; Head: Matrix }
    let init vocab dim hidden seed =
        let rng = Random(seed)
        let make r c = Array2D.init r c (fun _ _ -> (rng.NextDouble() * 2.0 - 1.0) * 0.15)
        { Emb = make vocab dim; Wq = make dim dim; Wk = make dim dim
          Wv = make dim dim; Wo = make dim dim; W1 = make dim hidden
          W2 = make hidden dim; Head = make dim vocab }
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
    open System.Security.Cryptography
    let private magic = 0x424C4D31 // BLM1
    let private version = 2
    let private maxFileBytes = 50L * 1024L * 1024L // 50 MiB hard cap
    let private matrices (w: Weights) = [|w.Emb; w.Wq; w.Wk; w.Wv; w.Wo; w.W1; w.W2; w.Head|]
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
        if not (storedHash.SequenceEqual(computed)) then failwith "checkpoint digest mismatch"
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
                match BigInteger.TryParse(t) with
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

module Checks =
    open Tensor
    let assertClose a b = if abs (a-b) > 1e-10 then failwithf "expected %g got %g" b a
    let run () =
        let a = array2D [[1.0;2.0];[3.0;4.0]]
        let b = array2D [[5.0;6.0];[7.0;8.0]]
        let c = matmul a b
        assertClose c[0,0] 19.0
        assertClose c[1,1] 50.0
        let p = softmaxRowsCausal (array2D [[1.0;2.0];[3.0;4.0]])
        assertClose p[0,0] 1.0
        assertClose p[0,1] 0.0
        assertClose (p[1,0]+p[1,1]) 1.0
        let w = Model.init 128 16 32 42
        let result = Model.forward w [|65;66;67|]
        if rows result <> 3 || cols result <> 128 then failwith "incorrect output shape"
        let short = Model.forward w [|65;66|]
        for j in 0..127 do assertClose short[1,j] result[1,j]
        let checkpoint = Path.Combine(Path.GetTempPath(), "bootleglm-" + Guid.NewGuid().ToString("N") + ".bin")
        try
            Checkpoint.save checkpoint w
            let restored = Checkpoint.load checkpoint
            let after = Model.forward restored [|65;66;67|]
            for i in 0..2 do for j in 0..127 do
                if result[i,j] <> after[i,j] then failwith "checkpoint roundtrip changed logits"
            use stream = new FileStream(checkpoint, FileMode.Append)
            stream.WriteByte(0uy)
            stream.Close()
            let mutable rejected = false
            try Checkpoint.load checkpoint |> ignore with _ -> rejected <- true
            if not rejected then failwith "trailing checkpoint data accepted"
        finally if File.Exists(checkpoint) then File.Delete(checkpoint)
        let expr = Reasoning.parse "(a+0)*(b*1)"
        match Reasoning.search 100 expr with
        | Some(normal,trace) ->
            if Reasoning.show normal <> "(a*b)" then failwith "wrong normal form"
            for ai in -3..3 do for bi in -3..3 do
                let a, b = bigint ai, bigint bi
                let env = function "a" -> a | "b" -> b | _ -> failwith "unbound"
                if Reasoning.evaluate env expr <> Reasoning.evaluate env normal then failwith "rewrite not equivalent"
            if trace.Length <> 2 then failwith "unexpected search length"
            if not (Reasoning.verifyTrace expr normal trace) then failwith "invalid rewrite certificate"
            let tampered = { trace.Head with After = Reasoning.Const 999I } :: trace.Tail
            if Reasoning.verifyTrace expr normal tampered then failwith "tampered proof accepted"
            let large = Reasoning.parse "99999999999999999999999999999+1"
            match Reasoning.search 20 large with
            | Some(result, proof) when Reasoning.show result = "100000000000000000000000000000" && Reasoning.verifyTrace large result proof -> ()
            | _ -> failwith "arbitrary precision rewrite failed"
        | None -> failwith "search failed"
        printfn "PASS: matmul, causal softmax, shapes, prefix invariance, checkpoint roundtrip/rejection, BFS rewriting, checked certificates, big integers"

[<EntryPoint>]
let main argv =
    try
        if argv.Length > 0 && argv[0] = "test" then
            Checks.run()
            0
        elif argv.Length > 0 && argv[0] = "save" then
            if argv.Length <> 2 then failwith "usage: save <checkpoint.bin>"
            Checkpoint.save argv[1] (Model.init 128 16 32 42)
            printfn "Saved checkpoint: %s" argv[1]
            0
        elif argv.Length > 0 && argv[0] = "load" then
            if argv.Length < 3 then failwith "usage: load <checkpoint.bin> <prompt>"
            let w = Checkpoint.load argv[1]
            let tokens = String.Join(" ",argv[2..]).ToCharArray() |> Array.map int
            let result = Model.forward w tokens
            printfn "Next token: %d" (Model.nextToken result)
            0
        elif argv.Length > 0 && argv[0] = "reason" then
            if argv.Length < 2 then failwith "usage: reason <expression>"
            let expr = Reasoning.parse (String.Join(" ",argv[1..]))
            match Reasoning.search 10000 expr with
            | None -> printfn "UNKNOWN: search exhausted"; 2
            | Some(normal,trace) ->
                if not (Reasoning.verifyTrace expr normal trace) then failwith "invalid search certificate"
                for step in trace do printfn "%A: %s -> %s" step.Rule (Reasoning.show step.Before) (Reasoning.show step.After)
                printfn "CERTIFICATE: verified %d steps" trace.Length
                printfn "RESULT: %s" (Reasoning.show normal)
                0
        elif argv.Length > 0 && argv[0] = "bench" then
            let w = Model.init 128 16 32 42
            let tokens = [|65;66;67;68;69;70;71;72|]
            for _ in 1..20 do Model.forward w tokens |> ignore // JIT + cache warmup
            let runCount = 200
            let sw = Diagnostics.Stopwatch.StartNew()
            let mutable digest = 0.0
            for _ in 1..runCount do
                let logits = Model.forward w tokens
                digest <- digest + logits[7,0]
            sw.Stop()
            let ms = sw.Elapsed.TotalMilliseconds / float runCount
            printfn "benchmark,os=%s,runtime=%s,iterations=%d,seq=%d,vocab=%d,dim=%d,hidden=%d,total_ms=%.6f,ms_per_forward=%.6f,checksum=%.17g" Environment.OSVersion.VersionString System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription runCount tokens.Length 128 16 32 sw.Elapsed.TotalMilliseconds ms digest
            0
        elif argv.Length > 0 && argv[0] = "rewrite" then
            if argv.Length < 3 then failwith "usage: rewrite <rules.roff> <expression>"
            let rules = Symbolic.load argv[1]
            let result, steps, limit = Symbolic.rewrite rules (String.Join(" ", argv[2..]))
            printfn "%s (passes=%d, limitReached=%b)" result steps limit
            0
        else
            let prompt = if argv.Length = 0 then "BOOTLEG" else String.Join(" ", argv)
            let w = Model.init 128 16 32 42
            let tokens = prompt.ToCharArray() |> Array.map (fun c -> int c % 128)
            let mutable context = tokens
            let generated = ResizeArray<char>()
            for _ in 1..16 do
                let next = Model.forward w context |> Model.nextToken
                generated.Add(char next)
                context <- Array.append context [|next|]
            printfn "Input: %s" prompt
            printfn "Generated token IDs: %s" (generated |> Seq.map (int >> string) |> String.concat " ")
            printfn "WARNING: deterministic random weights; output is not trained language."
            0
    with ex ->
        eprintfn "ERROR: %s" ex.Message
        1
