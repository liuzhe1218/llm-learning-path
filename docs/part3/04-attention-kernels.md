# 3.4 Attention Kernels

!!! abstract "What you'll learn"
    - Why a straightforward attention implementation is **memory-bound**, even though attention does a lot of arithmetic.
    - Derive the **online softmax** and implement **FlashAttention**'s tiled algorithm in NumPy, checking that it matches naive attention exactly.
    - Explain what FlashAttention-2 and -3 changed, at intuition level.
    - Explain why decode-time attention needs a different parallelization (**Flash-Decoding**, split-KV) and how **FlashInfer** handles paged and ragged KV layouts.
    - Understand **kernel launch overhead**, **CUDA graphs**, and **kernel fusion**.

## Why this matters / the problem

A GPU **kernel** is a function launched on the GPU; every operation in a PyTorch forward pass (a matmul, a softmax, an add) is one or more kernels. Each kernel typically reads its inputs from HBM and writes its outputs back to HBM. Chapter 3.1 showed that HBM bandwidth is the scarce resource. So the way operations are *packaged into kernels* — what stays in fast on-chip SRAM and what round-trips through HBM — can matter as much as the number of FLOPs.

Attention is the most important example. Written the obvious way, it materializes an $N \times N$ score matrix in HBM for an $N$-token sequence. That costs memory quadratic in context length (it was a hard limit on context length before 2022) and turns a FLOP-heavy operation into a memory-bound one. FlashAttention computes **exactly the same result** without ever writing that matrix to HBM. Every serving engine's attention backend (FlashAttention, FlashInfer, TensorRT-LLM's kernels, FlashMLA) is built on its ideas.

## Intuition

Imagine summing a weighted average over a book too large to fit on your desk. The naive way: photocopy every page (write $S$), go back and annotate every copy with a weight (softmax, write $P$), then read them all again to compute the average. The smart way: bring a few pages at a time to your desk, keep a **running** weighted sum and a **running** total weight, and never make copies. The only difficulty is that softmax weights depend on the **maximum** score over the whole row, which you do not know until you have seen every page. The **online softmax** trick fixes that: whenever you discover a larger maximum, rescale what you have accumulated so far.

## How it works

### Naive attention and its memory traffic

For one head, with $Q, K, V \in \mathbb{R}^{N \times d}$ ($N$ tokens, head dimension $d$):

$$
S = \frac{QK^\top}{\sqrt{d}} \in \mathbb{R}^{N \times N}, \qquad P = \operatorname{softmax}_{\text{row}}(S), \qquad O = PV \in \mathbb{R}^{N \times d}.
$$

A naive implementation runs three kernels: matmul (writes $S$ to HBM), softmax (reads $S$, writes $P$), matmul (reads $P$). The FLOPs are $2N^2d$ for $QK^\top$ plus $2N^2d$ for $PV$, i.e. $4N^2d$. The HBM traffic is dominated by $S$ and $P$: at least $4 \cdot N^2 \cdot b$ bytes (write and read each, $b$ bytes per element). The arithmetic intensity is therefore about

$$
I_{\text{naive}} \approx \frac{4N^2 d}{4N^2 b} = \frac{d}{b},
$$

which for $d=128$ in BF16 ($b=2$) is **64 FLOP/byte** — well below the H100's ≈ 295 FLOP/byte ridge point. Naive attention is memory-bound, and its intermediate memory grows as $N^2$.

### Online softmax

Softmax of a row $s_1, \dots, s_N$ is computed stably as $p_j = e^{s_j - m}/\ell$ with $m = \max_j s_j$ and $\ell = \sum_j e^{s_j - m}$ (subtracting $m$ prevents overflow and does not change the result). We want the output $o = \sum_j p_j v_j$ while visiting the scores in **blocks**.

Keep three running quantities after processing the first blocks: the running max $m$, the running denominator $\ell = \sum_{\text{seen}} e^{s_j - m}$, and the **unnormalized** output $\tilde o = \sum_{\text{seen}} e^{s_j - m} v_j$. When a new block with scores $s_{j}$, $j \in \text{block}$, arrives:

$$
\begin{aligned}
m' &= \max\big(m,\ \max_{j \in \text{block}} s_j\big),\\
\ell' &= e^{m - m'}\,\ell + \sum_{j \in \text{block}} e^{s_j - m'},\\
\tilde o' &= e^{m - m'}\,\tilde o + \sum_{j \in \text{block}} e^{s_j - m'}\, v_j .
\end{aligned}
$$

In words: everything accumulated so far was computed relative to the old max $m$; multiplying by $e^{m-m'}$ converts it to be relative to the new max $m'$ (because $e^{s-m}\cdot e^{m-m'} = e^{s-m'}$). After the last block, $o = \tilde o / \ell$ is **exactly** the softmax-weighted average — not an approximation. (Induction on the number of blocks: if $\ell$ and $\tilde o$ equal the full sums relative to $m$ after $k$ blocks, the update makes them equal the sums relative to $m'$ after $k+1$ blocks.)

The same algebra lets us **merge two partial results** computed independently on disjoint key sets: given $(m_1, \ell_1, \tilde o_1)$ and $(m_2, \ell_2, \tilde o_2)$, set $m = \max(m_1, m_2)$, $\ell = e^{m_1-m}\ell_1 + e^{m_2-m}\ell_2$, $\tilde o = e^{m_1-m}\tilde o_1 + e^{m_2-m}\tilde o_2$. This merge operator is the basis of Flash-Decoding and of cascade attention below.

### FlashAttention: tiling

FlashAttention splits $Q$ into row blocks of size $B_r$ and $K, V$ into blocks of size $B_c$, chosen so the tiles fit in an SM's shared memory. For each $Q$ block (in parallel across SMs):

1. Load the $Q$ tile into SRAM; initialize $m = -\infty$, $\ell = 0$, $\tilde O = 0$ for its rows.
2. For each $K, V$ block: load the tiles into SRAM, compute the $B_r \times B_c$ score tile $S_{ij} = Q_i K_j^\top/\sqrt d$ **in SRAM/registers**, apply the online-softmax update.
3. Write $O_i = \tilde O / \ell$ (and $m + \log \ell$, the log-sum-exp, which the backward pass needs) to HBM.

The $N \times N$ matrices never touch HBM. HBM traffic for $Q$ and $O$ is read/written once; $K$ and $V$ are streamed once per $Q$ block (and those repeated reads are often served by the L2 cache). The FlashAttention paper shows the HBM accesses are $\Theta(N^2 d^2 / M)$ for SRAM size $M$, versus $\Theta(Nd + N^2)$ for the standard implementation, and extra memory is $O(N)$ instead of $O(N^2)$. For causal attention, tiles entirely above the diagonal are skipped, halving the work.

### FlashAttention-2 and -3

- **FlashAttention-2** keeps the algorithm and reorganizes the work: fewer non-matmul operations (rescale the accumulator once at the end instead of every block), parallelism across the **sequence** dimension as well as batch and heads (so long sequences with small batch still fill the GPU), and a better split of work between warps inside a thread block that avoids communication through shared memory.
- **FlashAttention-3** targets Hopper GPUs (H100). Hopper can copy tiles between HBM and shared memory asynchronously (the TMA unit) and run tensor-core matmuls asynchronously (WGMMA instructions). FA-3 uses **warp specialization** (some warps only load data, others only compute) and overlaps the softmax of one tile with the matmuls of another, so the slow exponentials hide behind tensor-core work. It also adds FP8 attention, using block quantization and an "incoherent processing" trick (a random orthogonal transform) to reduce quantization error from outliers. The paper reports up to ~75% of the H100's peak BF16 throughput.

### Decode attention and Flash-Decoding

During decode each sequence contributes **one** query per head, attending to its $L$ cached keys. FlashAttention parallelizes over (batch × heads × query blocks). With batch 1, 8 KV heads and one query block, that is only a handful of thread blocks — on an H100 with 132 SMs, most of the GPU sits idle while one SM per head streams through a long KV cache.

**Flash-Decoding** adds parallelism along the **key** dimension: split the KV cache into $s$ chunks, have separate thread blocks compute partial results $(m_c, \ell_c, \tilde o_c)$ for each chunk in parallel, then **merge** them with the log-sum-exp merge above in a small reduction kernel. With $s = 16$, the same problem launches 16× more thread blocks and finishes up to several times faster at long context. Since decode attention is memory-bound anyway (Chapter 3.1), the goal is to have enough SMs issuing loads to saturate HBM bandwidth.

GQA helps here too: all query heads in a group can be processed together against one shared KV head, so each loaded K/V tile is reused $g$ times — the arithmetic intensity of decode attention rises from ≈ 1 to ≈ $g$ FLOP/byte.

### Paged and ragged layouts: FlashInfer

Real engines do not have neat `[batch, seq, heads, d]` tensors. With continuous batching (Chapter 3.3) each step contains sequences of different lengths, some in prefill (many queries) and some in decode (one query), and with PagedAttention their K/V live in scattered blocks found through block tables.

- **Ragged tensors**: concatenate all sequences' tokens without padding and pass an offsets array (`indptr`) marking where each sequence starts.
- **Paged KV**: pass the block table (page indices per sequence) and the length of the last page; the kernel gathers K/V tiles from those pages.

**FlashInfer** is a kernel library built for this world. It represents paged and ragged KV uniformly as block-sparse matrices, generates specialized kernels just-in-time for each attention variant (head dims, masks, logits soft-capping, sliding windows), and plans a load-balanced work split on the CPU before each step so that long and short sequences are spread evenly across SMs. It also implements **cascade attention**: when many requests share a prefix, attention over the shared prefix is computed once for all of their queries and merged with each request's own-suffix attention using the same merge operator. vLLM and SGLang use FlashInfer as one of their attention backends; DeepSeek's **FlashMLA** plays the same role for MLA decode.

### Kernel launch overhead and CUDA graphs

Launching a kernel from the CPU costs on the order of a few microseconds. A decode step of a transformer with 32–80 layers launches hundreds to thousands of kernels (norms, projections, RoPE, attention, activation, all-reduces…). For a small model or small batch, where each kernel runs only microseconds, the CPU cannot launch kernels fast enough and the GPU idles between them — the step becomes **launch-bound**.

**CUDA graphs** fix this: record the whole sequence of kernel launches once, then replay it with a single launch. The constraint is that the graph is fixed: tensor shapes and memory addresses must be the same on every replay. Engines therefore capture one graph per batch size in a set (e.g. 1, 2, 4, 8, 16, … up to some maximum), copy each step's inputs into the static input buffers, and pad the batch up to the nearest captured size. Prefill, with variable shapes, usually runs without graphs (it is long enough that launch overhead does not matter). `torch.compile` with `mode="reduce-overhead"` uses CUDA graphs automatically.

### Kernel fusion

Element-wise operations (add, multiply, activation, normalization) do ~1 FLOP per element read, so as separate kernels each one is a full HBM round-trip of the activation tensor. **Fusing** a chain of them into one kernel reads the input once and writes the output once. Standard fusions in LLM serving:

- residual add + RMSNorm in one kernel;
- SwiGLU: $\text{SiLU}(x W_1) \odot (x W_3)$ computed as one wide matmul plus a fused activation-and-multiply;
- QKV projection as one matmul, with RoPE applied inside the attention or a fused kernel;
- "epilogue fusion": applying bias, activation, or quantization in the matmul kernel before writing results;
- fused MoE kernels that route, gather tokens, and run all experts' matmuls in one grouped GEMM.

## Code

NumPy reference implementations: naive attention, tiled FlashAttention with online softmax (causal), and Flash-Decoding with split-KV and a log-sum-exp merge. All three must agree to floating-point precision.

```python
import numpy as np
rng = np.random.default_rng(0)

def naive_attention(Q, K, V, causal=True):
    N, d = Q.shape
    S = Q @ K.T / np.sqrt(d)                              # [N, N] materialized
    if causal:
        S = np.where(np.tril(np.ones((N, len(K)), bool), k=len(K) - N), S, -np.inf)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P / P.sum(axis=1, keepdims=True)) @ V

def flash_attention(Q, K, V, Br=16, Bc=16):
    """Causal attention, processed in [Br x Bc] tiles; never builds the N x N matrix."""
    N, d = Q.shape
    O = np.zeros_like(Q)
    for i in range(0, N, Br):                             # each Q block = one "thread block"
        q = Q[i:i + Br]
        m = np.full(len(q), -np.inf)                      # running max      [Br]
        l = np.zeros(len(q))                              # running sum      [Br]
        acc = np.zeros((len(q), d))                       # unnormalized out [Br, d]
        for j in range(0, i + len(q), Bc):                # skip tiles above the diagonal
            s = q @ K[j:j + Bc].T / np.sqrt(d)            # score tile [Br, Bc] (in "SRAM")
            rows, cols = np.arange(i, i + len(q))[:, None], np.arange(j, j + s.shape[1])
            s = np.where(cols <= rows, s, -np.inf)        # causal mask inside the tile
            m_new = np.maximum(m, s.max(axis=1))
            p = np.exp(s - m_new[:, None])
            scale = np.exp(m - m_new)                     # rescale old accumulators
            l = scale * l + p.sum(axis=1)
            acc = scale[:, None] * acc + p @ V[j:j + Bc]
            m = m_new
        O[i:i + Br] = acc / l[:, None]
    return O

def flash_decoding(q, K, V, n_splits=4):
    """One decode query q:[d] against L cached keys, KV split into chunks in parallel."""
    d = q.shape[0]
    parts = []
    for Kc, Vc in zip(np.array_split(K, n_splits), np.array_split(V, n_splits)):
        s = Kc @ q / np.sqrt(d)                           # each chunk: independent block
        m = s.max(); p = np.exp(s - m)
        parts.append((m, p.sum(), p @ Vc))                # (max, sum, unnormalized out)
    m = max(pm for pm, _, _ in parts)                     # log-sum-exp merge (reduction)
    l = sum(np.exp(pm - m) * pl for pm, pl, _ in parts)
    o = sum(np.exp(pm - m) * po for pm, _, po in parts)
    return o / l

N, d = 100, 64
Q, K, V = (rng.standard_normal((N, d)) for _ in range(3))
print("flash vs naive  :", np.abs(flash_attention(Q, K, V) - naive_attention(Q, K, V)).max())
L = 1000
Kc, Vc, q = rng.standard_normal((L, d)), rng.standard_normal((L, d)), rng.standard_normal(d)
ref = naive_attention(q[None], Kc, Vc, causal=False)[0]
print("decode vs naive :", np.abs(flash_decoding(q, Kc, Vc, n_splits=8) - ref).max())
```

How an engine uses CUDA graphs for decode with PyTorch (needs a CUDA GPU; the pattern — static buffers, warm-up, capture, copy-in, replay — is what vLLM and SGLang do per batch size):

```python
import torch

if torch.cuda.is_available():
    model = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024) for _ in range(40)]).cuda().half()
    static_in = torch.zeros(8, 1024, device="cuda", dtype=torch.half)  # batch size 8 graph

    s = torch.cuda.Stream()                    # warm up on a side stream before capture
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            model(static_in)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_out = model(static_in)          # 40 kernel launches recorded once

    new_batch = torch.randn(8, 1024, device="cuda", dtype=torch.half)
    static_in.copy_(new_batch)                 # write inputs into the captured buffer
    graph.replay()                             # one launch replays all 40 kernels
    print(torch.allclose(static_out, model(new_batch), atol=1e-2))
```

## Worked example

**Naive vs flash for one head.** $N = 8{,}192$ tokens, $d = 128$, BF16 ($b = 2$).

- FLOPs: $4N^2d = 4 \times 8192^2 \times 128 \approx 3.4\times10^{10}$.
- Naive intermediate traffic: $S$ and $P$ are each $N^2 = 6.7\times10^7$ elements = 128 MiB. Writing and reading both: $\approx 537$ MB. Intensity $\approx 3.4\times10^{10} / 5.4\times10^8 \approx 64$ FLOP/byte → memory-bound on an H100 (ridge ≈ 295). With 32 heads and a batch of 8, the score matrices alone would need $256 \times 128\ \text{MiB} = 32$ GiB of scratch memory.
- FlashAttention: the compulsory traffic is reading $Q, K, V$ and writing $O$: $4 \times 8192 \times 128 \times 2$ B = 8 MiB per head, and no $N^2$ scratch memory. $K$ and $V$ are re-streamed once per $Q$ block, but those reads are much cheaper than materializing $S$ and $P$ and are often served from L2. The kernel becomes compute-bound, which is why FA-3 can reach a large fraction of tensor-core peak.

**Flash-Decoding occupancy.** Batch 1, Llama-3-8B (32 query heads, 8 KV heads), context 32K, H100 with 132 SMs. Processing each KV head's query group in one thread block gives 8 blocks: ~6% of SMs busy. Splitting each head's 32K keys into 16 chunks of 2K gives $8 \times 16 = 128$ blocks — nearly one per SM — so all SMs stream KV from HBM concurrently. The merge kernel touches only $8 \times 16$ small partial results.

**Launch overhead.** Suppose a decode step launches ~10 kernels per layer for a 32-layer model plus a few extra: ~330 launches. At ~4 µs each that is ~1.3 ms of CPU launch time, compared with a bandwidth-bound step of ~4.8 ms for an 8B BF16 model on an H100 (Chapter 3.1) — a large overhead, and the dominant cost for a 1B model. A CUDA graph replay reduces it to a single launch. (Exact per-launch costs depend on CPU, driver and framework; the order of magnitude is the point.)

## In practice (2025–2026)

- FlashAttention-2/3 and FlashInfer are the default attention backends in open engines; Blackwell GPUs get their own kernels (e.g. FlashInfer and TensorRT-LLM ship Blackwell-specific attention, and FA-3-style ideas carry over).
- MLA models use dedicated decode kernels such as **FlashMLA**, because the latent-KV layout differs from GQA.
- All production engines use CUDA graphs for decode and capture a ladder of batch sizes at startup (this is part of why startup takes tens of seconds); vLLM V1 additionally uses `torch.compile` to fuse elementwise ops.
- Long-context decode uses split-KV automatically: kernels choose the number of splits from batch size, head count and context length.
- Kernel DSLs such as Triton and CUTLASS/CuTe (and newer Python DSLs) are how most of these kernels are written; the GPU MODE lectures are a good entry point.

## Common pitfalls / misconceptions

- **"FlashAttention is an approximation."** It is exact (up to floating-point reordering). Sparse/linear attention methods are approximations; FlashAttention is not.
- **"FlashAttention reduces FLOPs."** It performs the same (or slightly more, in the backward pass via recomputation) FLOPs; it reduces **HBM traffic and memory**.
- **Using a prefill-tuned kernel for decode.** Without split-KV, long-context, small-batch decode leaves most SMs idle.
- **Changing shapes under a CUDA graph.** Replaying a graph with tensors that were reallocated (new addresses) silently reads stale memory. Always copy into the captured static buffers.
- **Forgetting that split-KV changes floating-point summation order.** Results can differ in the last bits across batch sizes and split counts — relevant for determinism (Chapter 3.8).

## Key takeaways

- Naive attention writes $N \times N$ intermediates to HBM: arithmetic intensity ≈ $d/b$ (≈ 64 FLOP/byte for $d=128$, BF16) → memory-bound, and $O(N^2)$ memory.
- Online softmax keeps a running max, sum and unnormalized output, rescaling by $e^{m - m'}$; results are exact, and partial results merge with log-sum-exp.
- FlashAttention tiles Q/K/V through SRAM so scores never reach HBM; FA-2 improves parallelism and work partitioning; FA-3 exploits Hopper asynchrony and adds FP8.
- Decode attention needs parallelism over keys (Flash-Decoding) and support for paged/ragged layouts (FlashInfer).
- CUDA graphs remove kernel launch overhead in decode; fusion removes HBM round trips between element-wise ops.

## Check your understanding

??? question "In the online softmax, the running max increases from 3 to 5. By what factor are the old ℓ and õ multiplied, and why?"
    By $e^{3-5} = e^{-2}$. Old terms were stored as $e^{s - 3}$; to express them relative to the new max they must become $e^{s-5} = e^{s-3} e^{-2}$.

??? question "Why does FlashAttention's HBM traffic not depend on N² while naive attention's does?"
    The $N \times N$ score and probability tiles live only in SRAM/registers and are consumed immediately; only $Q, K, V$ (read) and $O$ (written) touch HBM, which are $O(Nd)$ each (plus re-streaming of K/V tiles, which depends on the SRAM size).

??? question "Why is Flash-Decoding unnecessary at batch size 256?"
    With 256 sequences × 8 KV heads = 2,048 independent (sequence, head) units, there is already far more parallelism than SMs. Splitting KV would only add merge overhead.

??? question "Why can't prefill use the same CUDA graph for every request?"
    Prefill shapes depend on prompt length, which varies per request; graphs require fixed shapes and addresses. Prefill kernels are also long enough that launch overhead is negligible.

??? question "What makes cascade attention possible mathematically?"
    The merge operator for partial softmax results: attention over (shared prefix ∪ own suffix) equals the log-sum-exp merge of attention over the prefix and attention over the suffix, so the prefix part can be computed once for many queries.

## Further reading

| Paper | Why read it |
|---|---|
| :material-star: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 | The tiling + online-softmax algorithm and its IO-complexity analysis. |
| :material-book-open-variant: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023 | Sequence-dimension parallelism and better warp-level work split. |
| :material-new-box: [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Shah et al., 2024 | Hopper warp specialization, softmax/matmul overlap, FP8 attention. |
| :material-book-open-variant: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — Milakov & Gimelshein, 2018 | The original online-softmax derivation that FlashAttention builds on. |
| :material-book-open-variant: [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) — Dao et al., 2023 (blog) | Split-KV decode attention so long-context, small-batch decode uses the whole GPU. |
| :material-book-open-variant: [FlashInfer: Efficient and Customizable Attention Engine for LLM Serving](https://arxiv.org/abs/2501.01005) — Ye et al., MLSys 2025 | JIT attention kernels for paged/ragged KV, load-balanced scheduling, cascade attention; used in vLLM and SGLang. |
| :material-book-open-variant: [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | DeepSeek's decode kernels for multi-head latent attention. |
| :material-book-open-variant: [GPU MODE lectures](https://github.com/gpu-mode/lectures) | Community lectures on CUDA, Triton and writing kernels like these. |
