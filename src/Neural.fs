module Neural

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
open System.Security.Cryptography
open System.Text

module Stack =
    // Multi-head causal transformer.
    // Each block is layer-norm, causal multi-head attention, residual,
    // layer-norm, ReLU feed-forward, residual. A final layer-norm feeds the token head.
    // BLM1 version 3 stores this network. Version 2 remains the single-head block.
    let version = 3
    let eps = 1e-5
    let initScale = 0.05
    let private magic = 0x424C4D31
    let private maxFileBytes = 50L * 1024L * 1024L
    type Config = { Vocab: int; Dim: int; Heads: int; Layers: int; Hidden: int }
    type Layer = {
        Wq: float[,]
        Wk: float[,]
        Wv: float[,]
        Wo: float[,]
        W1: float[,]
        W2: float[,]
        Ln1G: float[]
        Ln1B: float[]
        Ln2G: float[]
        Ln2B: float[]
    }
    type Net = {
        Config: Config
        Emb: float[,]
        Layers: Layer[]
        LnFG: float[]
        LnFB: float[]
        Head: float[,]
    }
    type Grads = {
        Emb: float[,]
        Layers: Layer[]
        LnFG: float[]
        LnFB: float[]
        Head: float[,]
    }
    let defaultConfig = { Vocab = 128; Dim = 16; Heads = 4; Layers = 2; Hidden = 32 }
    let defaultSeed = 42
    let headDim (cfg: Config) =
        if cfg.Vocab < 2 || cfg.Dim < 1 || cfg.Heads < 1 || cfg.Layers < 1 || cfg.Hidden < 1 then
            invalidArg "config" "network dimensions must be positive"
        if cfg.Dim % cfg.Heads <> 0 then invalidArg "config" "dimension must be divisible by the head count"
        cfg.Dim / cfg.Heads
    let private rows (a: float[,]) = a.GetLength 0
    let private cols (a: float[,]) = a.GetLength 1
    let private makeMatrix (rng: Random) r c =
        Array2D.init r c (fun _ _ -> (rng.NextDouble() * 2.0 - 1.0) * initScale)
    let private ones n = Array.create n 1.0
    let private zeros n = Array.zeroCreate<float> n
    let private zeroMatrix r c = Array2D.zeroCreate<float> r c
    let private layerOf (rng: Random) (cfg: Config) =
        { Wq = makeMatrix rng cfg.Dim cfg.Dim
          Wk = makeMatrix rng cfg.Dim cfg.Dim
          Wv = makeMatrix rng cfg.Dim cfg.Dim
          Wo = makeMatrix rng cfg.Dim cfg.Dim
          W1 = makeMatrix rng cfg.Dim cfg.Hidden
          W2 = makeMatrix rng cfg.Hidden cfg.Dim
          Ln1G = ones cfg.Dim
          Ln1B = zeros cfg.Dim
          Ln2G = ones cfg.Dim
          Ln2B = zeros cfg.Dim }
    let init (cfg: Config) seed =
        let _ = headDim cfg
        let rng = Random(seed)
        { Config = cfg
          Emb = makeMatrix rng cfg.Vocab cfg.Dim
          Layers = Array.init cfg.Layers (fun _ -> layerOf rng cfg)
          LnFG = ones cfg.Dim
          LnFB = zeros cfg.Dim
          Head = makeMatrix rng cfg.Dim cfg.Vocab }
    let initDefault () = init defaultConfig defaultSeed
    let private matmul (a: float[,]) (b: float[,]) =
        let m = rows a
        let k = cols a
        let n = cols b
        if k <> rows b then invalidArg "b" "incompatible matrix dimensions"
        let y = zeroMatrix m n
        for i in 0 .. m - 1 do
            for p in 0 .. k - 1 do
                let aip = a[i, p]
                if aip <> 0.0 then
                    for j in 0 .. n - 1 do
                        y[i, j] <- y[i, j] + aip * b[p, j]
        y
    let private matmulBT (dy: float[,]) (b: float[,]) =
        let m = rows dy
        let n = cols dy
        let k = rows b
        let y = zeroMatrix m k
        for i in 0 .. m - 1 do
            for p in 0 .. k - 1 do
                let mutable s = 0.0
                for j in 0 .. n - 1 do s <- s + dy[i, j] * b[p, j]
                y[i, p] <- s
        y
    let private matmulAT (a: float[,]) (dy: float[,]) =
        let m = rows a
        let k = cols a
        let n = cols dy
        let y = zeroMatrix k n
        for i in 0 .. m - 1 do
            for p in 0 .. k - 1 do
                let aip = a[i, p]
                if aip <> 0.0 then
                    for j in 0 .. n - 1 do
                        y[p, j] <- y[p, j] + aip * dy[i, j]
        y
    let private copyMatrix (a: float[,]) =
        Array2D.init (rows a) (cols a) (fun i j -> a[i, j])
    let private addInto (dst: float[,]) (src: float[,]) =
        for i in 0 .. rows dst - 1 do
            for j in 0 .. cols dst - 1 do
                dst[i, j] <- dst[i, j] + src[i, j]
    let private layerNorm (x: float[,]) (g: float[]) (b: float[]) =
        let t = rows x
        let d = cols x
        let y = zeroMatrix t d
        let xhat = zeroMatrix t d
        let inv = zeros t
        for i in 0 .. t - 1 do
            let mutable mean = 0.0
            for j in 0 .. d - 1 do mean <- mean + x[i, j]
            mean <- mean / float d
            let mutable var = 0.0
            for j in 0 .. d - 1 do
                let diff = x[i, j] - mean
                var <- var + diff * diff
            var <- var / float d
            let invStd = 1.0 / sqrt (var + eps)
            inv[i] <- invStd
            for j in 0 .. d - 1 do
                let hat = (x[i, j] - mean) * invStd
                xhat[i, j] <- hat
                y[i, j] <- g[j] * hat + b[j]
        y, xhat, inv
    let private layerNormBackward (dy: float[,]) (xhat: float[,]) (inv: float[]) (g: float[]) =
        let t = rows dy
        let d = cols dy
        let dx = zeroMatrix t d
        let dg = zeros d
        let db = zeros d
        for j in 0 .. d - 1 do
            let mutable sg = 0.0
            let mutable sb = 0.0
            for i in 0 .. t - 1 do
                sg <- sg + dy[i, j] * xhat[i, j]
                sb <- sb + dy[i, j]
            dg[j] <- sg
            db[j] <- sb
        for i in 0 .. t - 1 do
            let dxh = zeros d
            let mutable s1 = 0.0
            let mutable s2 = 0.0
            for j in 0 .. d - 1 do
                let value = dy[i, j] * g[j]
                dxh[j] <- value
                s1 <- s1 + value
                s2 <- s2 + value * xhat[i, j]
            let scale = inv[i] / float d
            for j in 0 .. d - 1 do
                dx[i, j] <- scale * (float d * dxh[j] - s1 - xhat[i, j] * s2)
        dx, dg, db
    let private sliceCols (a: float[,]) col count =
        let y = zeroMatrix (rows a) count
        for i in 0 .. rows a - 1 do
            for j in 0 .. count - 1 do y[i, j] <- a[i, col + j]
        y
    let private addSlice (dst: float[,]) col (src: float[,]) =
        for i in 0 .. rows src - 1 do
            for j in 0 .. cols src - 1 do
                dst[i, col + j] <- dst[i, col + j] + src[i, j]
    let private softmaxCausal (scores: float[,]) =
        let t = rows scores
        let attn = zeroMatrix t t
        for i in 0 .. t - 1 do
            let mutable mx = Double.NegativeInfinity
            for j in 0 .. i do mx <- max mx scores[i, j]
            let mutable sum = 0.0
            for j in 0 .. i do
                let e = exp (scores[i, j] - mx)
                attn[i, j] <- e
                sum <- sum + e
            for j in 0 .. i do attn[i, j] <- attn[i, j] / sum
        attn
    type private HeadAct = { Q: float[,]; K: float[,]; V: float[,]; Attn: float[,] }
    type private LayerAct = {
        Input: float[,]
        N1: float[,]
        Xhat1: float[,]
        Inv1: float[]
        Heads: HeadAct[]
        Concat: float[,]
        H1: float[,]
        N2: float[,]
        Xhat2: float[,]
        Inv2: float[]
        Pre: float[,]
        Relu: float[,]
    }
    let private attend (x: float[,]) (layer: Layer) heads =
        let t = rows x
        let d = cols x
        let hd = d / heads
        let scale = sqrt (float hd)
        let qAll = matmul x layer.Wq
        let kAll = matmul x layer.Wk
        let vAll = matmul x layer.Wv
        let concat = zeroMatrix t d
        let acts = Array.init heads (fun h ->
            let col = h * hd
            let q = sliceCols qAll col hd
            let k = sliceCols kAll col hd
            let v = sliceCols vAll col hd
            let scores = zeroMatrix t t
            for i in 0 .. t - 1 do
                for j in 0 .. i do
                    let mutable dot = 0.0
                    for p in 0 .. hd - 1 do dot <- dot + q[i, p] * k[j, p]
                    scores[i, j] <- dot / scale
            let attn = softmaxCausal scores
            addSlice concat col (matmul attn v)
            { Q = q; K = k; V = v; Attn = attn })
        matmul concat layer.Wo, concat, acts
    let private embed (net: Net) (tokens: int[]) =
        let t = tokens.Length
        let x = zeroMatrix t net.Config.Dim
        for i in 0 .. t - 1 do
            let id = tokens[i]
            if id < 0 || id >= net.Config.Vocab then invalidArg "tokens" "out of vocabulary"
            for j in 0 .. net.Config.Dim - 1 do x[i, j] <- net.Emb[id, j]
        x
    let private run (net: Net) (tokens: int[]) =
        if tokens.Length < 2 then invalidArg "tokens" "need at least two tokens so a next token exists"
        let cfg = net.Config
        let mutable current = embed net tokens
        let acts = Array.init cfg.Layers (fun li ->
            let layer = net.Layers[li]
            let input = current
            let n1, xhat1, inv1 = layerNorm input layer.Ln1G layer.Ln1B
            let attn, concat, heads = attend n1 layer cfg.Heads
            let h1 = Array2D.init (rows input) cfg.Dim (fun i j -> input[i, j] + attn[i, j])
            let n2, xhat2, inv2 = layerNorm h1 layer.Ln2G layer.Ln2B
            let pre = matmul n2 layer.W1
            let rel = Array2D.init (rows pre) (cols pre) (fun i j -> max 0.0 pre[i, j])
            let ff = matmul rel layer.W2
            current <- Array2D.init (rows h1) cfg.Dim (fun i j -> h1[i, j] + ff[i, j])
            { Input = input; N1 = n1; Xhat1 = xhat1; Inv1 = inv1; Heads = heads; Concat = concat
              H1 = h1; N2 = n2; Xhat2 = xhat2; Inv2 = inv2; Pre = pre; Relu = rel })
        let final, xhatF, invF = layerNorm current net.LnFG net.LnFB
        let logits = matmul final net.Head
        acts, final, xhatF, invF, logits
    let forward (net: Net) (tokens: int[]) =
        if tokens.Length = 0 then invalidArg "tokens" "empty input"
        // A single token still has a distribution over the next id. Pad a dummy target
        // only inside training. Inference allows length 1 by reusing the same cache path
        // with a repeated token that is then truncated to the real rows.
        if tokens.Length = 1 then
            let padded = [| tokens[0]; tokens[0] |]
            let _, _, _, _, logits = run net padded
            Array2D.init 1 (cols logits) (fun _ j -> logits[0, j])
        else
            let _, _, _, _, logits = run net tokens
            logits
    let private crossEntropy (logits: float[,]) (tokens: int[]) =
        let t = rows logits
        let v = cols logits
        let npos = t - 1
        let dlogits = zeroMatrix t v
        let mutable loss = 0.0
        for i in 0 .. npos - 1 do
            let mutable mx = Double.NegativeInfinity
            for j in 0 .. v - 1 do mx <- max mx logits[i, j]
            let exps = zeros v
            let mutable sum = 0.0
            for j in 0 .. v - 1 do
                let e = exp (logits[i, j] - mx)
                exps[j] <- e
                sum <- sum + e
            let target = tokens[i + 1]
            let prob = exps[target] / sum
            loss <- loss - log (max prob 1e-12)
            for j in 0 .. v - 1 do dlogits[i, j] <- exps[j] / sum
            dlogits[i, target] <- dlogits[i, target] - 1.0
        loss <- loss / float npos
        let inv = 1.0 / float npos
        for i in 0 .. npos - 1 do
            for j in 0 .. v - 1 do dlogits[i, j] <- dlogits[i, j] * inv
        loss, dlogits
    let private zeroLayer (cfg: Config) =
        { Wq = zeroMatrix cfg.Dim cfg.Dim
          Wk = zeroMatrix cfg.Dim cfg.Dim
          Wv = zeroMatrix cfg.Dim cfg.Dim
          Wo = zeroMatrix cfg.Dim cfg.Dim
          W1 = zeroMatrix cfg.Dim cfg.Hidden
          W2 = zeroMatrix cfg.Hidden cfg.Dim
          Ln1G = zeros cfg.Dim
          Ln1B = zeros cfg.Dim
          Ln2G = zeros cfg.Dim
          Ln2B = zeros cfg.Dim }
    let private attendBackward (dAttn: float[,]) (layer: Layer) (act: LayerAct) heads =
        let dConcat = matmulBT dAttn layer.Wo
        let dWo = matmulAT act.Concat dAttn
        let t = rows act.N1
        let d = cols act.N1
        let hd = d / heads
        let scale = sqrt (float hd)
        let dQ = zeroMatrix t d
        let dK = zeroMatrix t d
        let dV = zeroMatrix t d
        for h in 0 .. heads - 1 do
            let col = h * hd
            let head = act.Heads[h]
            let dho = sliceCols dConcat col hd
            let dAttnHead = matmulBT dho head.V
            addSlice dV col (matmulAT head.Attn dho)
            let dScore = zeroMatrix t t
            for i in 0 .. t - 1 do
                let mutable dot = 0.0
                for j in 0 .. i do dot <- dot + head.Attn[i, j] * dAttnHead[i, j]
                for j in 0 .. i do
                    dScore[i, j] <- head.Attn[i, j] * (dAttnHead[i, j] - dot)
            for i in 0 .. t - 1 do
                for j in 0 .. i do dScore[i, j] <- dScore[i, j] / scale
            addSlice dQ col (matmul dScore head.K)
            addSlice dK col (matmulAT dScore head.Q)
        let dWq = matmulAT act.N1 dQ
        let dWk = matmulAT act.N1 dK
        let dWv = matmulAT act.N1 dV
        let dN1 = zeroMatrix t d
        addInto dN1 (matmulBT dQ layer.Wq)
        addInto dN1 (matmulBT dK layer.Wk)
        addInto dN1 (matmulBT dV layer.Wv)
        dN1, dWq, dWk, dWv, dWo
    let backward (net: Net) (tokens: int[]) =
        let cfg = net.Config
        let acts, final, xhatF, invF, logits = run net tokens
        let loss, dlogits = crossEntropy logits tokens
        let dFinal = matmulBT dlogits net.Head
        let dHead = matmulAT final dlogits
        let mutable dCurrent, dLnFG, dLnFB = layerNormBackward dFinal xhatF invF net.LnFG
        let layerGrads = Array.zeroCreate cfg.Layers
        for li in cfg.Layers - 1 .. -1 .. 0 do
            let layer = net.Layers[li]
            let act = acts[li]
            let dH1 = copyMatrix dCurrent
            let dFf = dCurrent
            let dRelu = matmulBT dFf layer.W2
            let dW2 = matmulAT act.Relu dFf
            let dPre = zeroMatrix (rows dRelu) (cols dRelu)
            for i in 0 .. rows dPre - 1 do
                for j in 0 .. cols dPre - 1 do
                    if act.Pre[i, j] > 0.0 then dPre[i, j] <- dRelu[i, j]
            let dN2 = matmulBT dPre layer.W1
            let dW1 = matmulAT act.N2 dPre
            let dx2, dLn2G, dLn2B = layerNormBackward dN2 act.Xhat2 act.Inv2 layer.Ln2G
            addInto dH1 dx2
            let dN1, dWq, dWk, dWv, dWo = attendBackward dH1 layer act cfg.Heads
            let dx1, dLn1G, dLn1B = layerNormBackward dN1 act.Xhat1 act.Inv1 layer.Ln1G
            dCurrent <- copyMatrix dH1
            addInto dCurrent dx1
            layerGrads[li] <-
                { Wq = dWq; Wk = dWk; Wv = dWv; Wo = dWo; W1 = dW1; W2 = dW2
                  Ln1G = dLn1G; Ln1B = dLn1B; Ln2G = dLn2G; Ln2B = dLn2B }
        let dEmb = zeroMatrix cfg.Vocab cfg.Dim
        for i in 0 .. tokens.Length - 1 do
            let id = tokens[i]
            for j in 0 .. cfg.Dim - 1 do
                dEmb[id, j] <- dEmb[id, j] + dCurrent[i, j]
        loss, { Emb = dEmb; Layers = layerGrads; LnFG = dLnFG; LnFB = dLnFB; Head = dHead }
    let private axpyMatrix lr (grad: float[,]) (param: float[,]) =
        for i in 0 .. rows param - 1 do
            for j in 0 .. cols param - 1 do
                param[i, j] <- param[i, j] - lr * grad[i, j]
    let private axpyVector lr (grad: float[]) (param: float[]) =
        for i in 0 .. param.Length - 1 do
            param[i] <- param[i] - lr * grad[i]
    let private applyLayer lr (grad: Layer) (param: Layer) =
        axpyMatrix lr grad.Wq param.Wq
        axpyMatrix lr grad.Wk param.Wk
        axpyMatrix lr grad.Wv param.Wv
        axpyMatrix lr grad.Wo param.Wo
        axpyMatrix lr grad.W1 param.W1
        axpyMatrix lr grad.W2 param.W2
        axpyVector lr grad.Ln1G param.Ln1G
        axpyVector lr grad.Ln1B param.Ln1B
        axpyVector lr grad.Ln2G param.Ln2G
        axpyVector lr grad.Ln2B param.Ln2B
    let sgdStep (net: Net) (tokens: int[]) lr =
        if lr <= 0.0 then invalidArg "lr" "learning rate must be positive"
        let loss, grads = backward net tokens
        if Double.IsNaN loss || Double.IsInfinity loss then failwith "training loss is not finite"
        axpyMatrix lr grads.Emb net.Emb
        for i in 0 .. net.Layers.Length - 1 do applyLayer lr grads.Layers[i] net.Layers[i]
        axpyVector lr grads.LnFG net.LnFG
        axpyVector lr grads.LnFB net.LnFB
        axpyMatrix lr grads.Head net.Head
        loss
    let loss (net: Net) (tokens: int[]) =
        let _, _, _, _, logits = run net tokens
        let value, _ = crossEntropy logits tokens
        value
    let train (net: Net) (tokens: int[]) steps seqLen lr =
        if steps < 1 then invalidArg "steps" "must be positive"
        if seqLen < 2 then invalidArg "seq" "sequence must contain a next token"
        if tokens.Length <= seqLen then invalidArg "tokens" "corpus shorter than the sequence"
        Array.init steps (fun step ->
            let start = (step * 3) % (tokens.Length - seqLen)
            let window = Array.sub tokens start seqLen
            sgdStep net window lr)
    let loadCorpus (path: string) vocab =
        let text = File.ReadAllText(path).Replace("\r", "")
        if text.Length < 2 then failwith "corpus too small"
        text.ToCharArray() |> Array.map (fun c -> int c % vocab)
    let private finiteMatrix (a: float[,]) =
        for i in 0 .. rows a - 1 do
            for j in 0 .. cols a - 1 do
                if Double.IsNaN a[i, j] || Double.IsInfinity a[i, j] then failwith "nonfinite weight"
    let private finiteVector (a: float[]) =
        for value in a do
            if Double.IsNaN value || Double.IsInfinity value then failwith "nonfinite weight"
    let private validate (net: Net) =
        let cfg = net.Config
        let hd = headDim cfg
        if hd * cfg.Heads <> cfg.Dim then failwith "checkpoint shape mismatch"
        if rows net.Emb <> cfg.Vocab || cols net.Emb <> cfg.Dim then failwith "checkpoint shape mismatch"
        if rows net.Head <> cfg.Dim || cols net.Head <> cfg.Vocab then failwith "checkpoint shape mismatch"
        if net.LnFG.Length <> cfg.Dim || net.LnFB.Length <> cfg.Dim then failwith "checkpoint shape mismatch"
        if net.Layers.Length <> cfg.Layers then failwith "checkpoint shape mismatch"
        finiteMatrix net.Emb
        finiteMatrix net.Head
        finiteVector net.LnFG
        finiteVector net.LnFB
        for layer in net.Layers do
            let expect (a: float[,]) r c =
                if rows a <> r || cols a <> c then failwith "checkpoint shape mismatch"
                finiteMatrix a
            expect layer.Wq cfg.Dim cfg.Dim
            expect layer.Wk cfg.Dim cfg.Dim
            expect layer.Wv cfg.Dim cfg.Dim
            expect layer.Wo cfg.Dim cfg.Dim
            expect layer.W1 cfg.Dim cfg.Hidden
            expect layer.W2 cfg.Hidden cfg.Dim
            if layer.Ln1G.Length <> cfg.Dim || layer.Ln1B.Length <> cfg.Dim ||
               layer.Ln2G.Length <> cfg.Dim || layer.Ln2B.Length <> cfg.Dim then failwith "checkpoint shape mismatch"
            finiteVector layer.Ln1G
            finiteVector layer.Ln1B
            finiteVector layer.Ln2G
            finiteVector layer.Ln2B
    let private writeMatrix (bw: BinaryWriter) (a: float[,]) =
        bw.Write(rows a)
        bw.Write(cols a)
        for i in 0 .. rows a - 1 do
            for j in 0 .. cols a - 1 do bw.Write(a[i, j])
    let private writeVector (bw: BinaryWriter) (a: float[]) =
        bw.Write(a.Length)
        for value in a do bw.Write(value)
    let private readMatrix (br: BinaryReader) =
        let r = br.ReadInt32()
        let c = br.ReadInt32()
        if r < 1 || c < 1 || int64 r * int64 c > 10000000L then failwith "invalid checkpoint matrix dimensions"
        Array2D.init r c (fun _ _ -> br.ReadDouble())
    let private readVector (br: BinaryReader) =
        let n = br.ReadInt32()
        if n < 1 || n > 10000000 then failwith "invalid checkpoint vector length"
        Array.init n (fun _ -> br.ReadDouble())
    let private payloadOf (net: Net) =
        validate net
        use ms = new MemoryStream()
        use bw = new BinaryWriter(ms, Encoding.UTF8, true)
        let cfg = net.Config
        bw.Write(magic)
        bw.Write(version)
        bw.Write(cfg.Vocab)
        bw.Write(cfg.Dim)
        bw.Write(cfg.Heads)
        bw.Write(cfg.Layers)
        bw.Write(cfg.Hidden)
        writeMatrix bw net.Emb
        for layer in net.Layers do
            writeMatrix bw layer.Wq
            writeMatrix bw layer.Wk
            writeMatrix bw layer.Wv
            writeMatrix bw layer.Wo
            writeMatrix bw layer.W1
            writeMatrix bw layer.W2
            writeVector bw layer.Ln1G
            writeVector bw layer.Ln1B
            writeVector bw layer.Ln2G
            writeVector bw layer.Ln2B
        writeVector bw net.LnFG
        writeVector bw net.LnFB
        writeMatrix bw net.Head
        bw.Flush()
        ms.ToArray()
    let private bytesEqual (a: byte[]) (b: byte[]) =
        a.Length = b.Length && Array.forall2 (fun x y -> x = y) a b
    let save path (net: Net) =
        let payload = payloadOf net
        if int64 payload.Length > maxFileBytes then failwith "checkpoint exceeds size cap"
        let hash = SHA256.HashData(payload)
        let tmp = path + ".tmp"
        try
            do
                use fs = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None)
                fs.Write(payload, 0, payload.Length)
                fs.Write(hash, 0, hash.Length)
                fs.Flush(true)
            File.Move(tmp, path, true)
        finally
            if File.Exists tmp then File.Delete tmp
    let load path =
        let info = FileInfo(path)
        if info.Length > maxFileBytes + 32L then failwith "checkpoint exceeds size cap"
        use fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read)
        use br = new BinaryReader(fs)
        let payloadLen = int (fs.Length - 32L)
        if payloadLen < 16 then failwith "checkpoint too small"
        let payload = br.ReadBytes(payloadLen)
        let stored = br.ReadBytes(32)
        if br.BaseStream.Position <> br.BaseStream.Length then failwith "checkpoint has trailing bytes"
        if not (bytesEqual stored (SHA256.HashData payload)) then failwith "checkpoint digest mismatch"
        use ms = new MemoryStream(payload)
        use pr = new BinaryReader(ms)
        if pr.ReadInt32() <> magic || pr.ReadInt32() <> version then failwith "invalid checkpoint header"
        let cfg = { Vocab = pr.ReadInt32(); Dim = pr.ReadInt32(); Heads = pr.ReadInt32(); Layers = pr.ReadInt32(); Hidden = pr.ReadInt32() }
        let emb = readMatrix pr
        let layers = Array.init cfg.Layers (fun _ ->
            { Wq = readMatrix pr; Wk = readMatrix pr; Wv = readMatrix pr; Wo = readMatrix pr
              W1 = readMatrix pr; W2 = readMatrix pr
              Ln1G = readVector pr; Ln1B = readVector pr; Ln2G = readVector pr; Ln2B = readVector pr })
        let net = { Config = cfg; Emb = emb; Layers = layers; LnFG = readVector pr; LnFB = readVector pr; Head = readMatrix pr }
        if ms.Position <> ms.Length then failwith "checkpoint payload has trailing data"
        validate net
        net
    let copy (net: Net) =
        let copyLayer (layer: Layer) =
            { Wq = copyMatrix layer.Wq; Wk = copyMatrix layer.Wk; Wv = copyMatrix layer.Wv; Wo = copyMatrix layer.Wo
              W1 = copyMatrix layer.W1; W2 = copyMatrix layer.W2
              Ln1G = Array.copy layer.Ln1G; Ln1B = Array.copy layer.Ln1B
              Ln2G = Array.copy layer.Ln2G; Ln2B = Array.copy layer.Ln2B }
        { net with
            Emb = copyMatrix net.Emb
            Layers = Array.map copyLayer net.Layers
            LnFG = Array.copy net.LnFG
            LnFB = Array.copy net.LnFB
            Head = copyMatrix net.Head }
