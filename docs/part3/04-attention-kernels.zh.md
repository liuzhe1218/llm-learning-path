# 3.4 Attention kernel

!!! abstract "本章你将学到"
    - 为什么直观的 attention 实现是**显存受限**的，尽管 attention 做了很多算术运算。
    - 推导**在线 softmax**，并用 NumPy 实现 **FlashAttention** 的分块算法，验证它与朴素 attention 完全一致。
    - 从直觉层面解释 FlashAttention-2 和 -3 改了什么。
    - 解释为什么解码时的 attention 需要不同的并行方式（**Flash-Decoding**，split-KV），以及 **FlashInfer** 如何处理分页和不规则（ragged）的 KV 布局。
    - 理解 **kernel 启动开销**、**CUDA graph** 和 **kernel 融合**。

## 为什么重要 {#why-this-matters-the-problem}

GPU **kernel** 是在 GPU 上启动的函数；PyTorch 前向计算中的每个操作（矩阵乘、softmax、加法）都对应一个或多个 kernel。每个 kernel 通常从 HBM 读入输入，再把输出写回 HBM。3.1 章说明了 HBM 带宽是稀缺资源。所以，操作如何被*打包进 kernel*——什么留在快速的片上 SRAM，什么要往返 HBM——与 FLOPs 的数量同样重要。

Attention 是最重要的例子。按最直观的方式实现，对于 $N$ token 的序列，它会在 HBM 中物化一个 $N \times N$ 的分数矩阵。这带来随上下文长度平方增长的内存（2022 年之前这是上下文长度的硬上限），并把一个 FLOP 密集的操作变成了显存受限的操作。FlashAttention 计算出**完全相同的结果**，却从不把那个矩阵写进 HBM。每个推理引擎的 attention 后端（FlashAttention、FlashInfer、TensorRT-LLM 的 kernel、FlashMLA）都建立在它的思想之上。

## 直觉 {#intuition}

想象要对一本大到桌子放不下的书做加权平均。朴素的做法：把每一页都复印一份（写出 $S$），回头在每份复印件上标注权重（softmax，写出 $P$），然后再把它们全部读一遍来计算平均值。聪明的做法：每次只把几页带到桌上，维护一个**滚动的**加权和和一个**滚动的**总权重，永远不复印。唯一的难点是 softmax 的权重依赖整行分数的**最大值**，而你在看完每一页之前并不知道它。**在线 softmax** 技巧解决了这个问题：每当发现更大的最大值，就把到目前为止累积的东西重新缩放。

## 工作原理 {#how-it-works}

### 朴素 attention 及其内存流量 {#naive-attention-and-its-memory-traffic}

对于一个头，$Q, K, V \in \mathbb{R}^{N \times d}$（$N$ 个 token，头维度 $d$）：

$$
S = \frac{QK^\top}{\sqrt{d}} \in \mathbb{R}^{N \times N}, \qquad P = \operatorname{softmax}_{\text{row}}(S), \qquad O = PV \in \mathbb{R}^{N \times d}.
$$

朴素实现会跑三个 kernel：矩阵乘（把 $S$ 写入 HBM）、softmax（读 $S$，写 $P$）、矩阵乘（读 $P$）。FLOPs 是 $QK^\top$ 的 $2N^2d$ 加上 $PV$ 的 $2N^2d$，即 $4N^2d$。HBM 流量主要来自 $S$ 和 $P$：至少 $4 \cdot N^2 \cdot b$ 字节（各写一次读一次，每个元素 $b$ 字节）。所以算术强度大约是

$$
I_{\text{naive}} \approx \frac{4N^2 d}{4N^2 b} = \frac{d}{b},
$$

对于 BF16（$b=2$）下的 $d=128$，这是 **64 FLOP/byte**——远低于 H100 约 295 FLOP/byte 的拐点。朴素 attention 是显存受限的，并且它的中间内存按 $N^2$ 增长。

### 在线 softmax {#online-softmax}

一行 $s_1, \dots, s_N$ 的 softmax 可以稳定地计算为 $p_j = e^{s_j - m}/\ell$，其中 $m = \max_j s_j$，$\ell = \sum_j e^{s_j - m}$（减去 $m$ 能防止溢出，且不改变结果）。我们想在按**块**访问分数的同时，算出输出 $o = \sum_j p_j v_j$。

在处理完前面若干块之后，维护三个滚动量：滚动最大值 $m$、滚动分母 $\ell = \sum_{\text{seen}} e^{s_j - m}$，以及**未归一化**的输出 $\tilde o = \sum_{\text{seen}} e^{s_j - m} v_j$。当一个新块的分数 $s_{j}$（$j \in \text{block}$）到来时：

$$
\begin{aligned}
m' &= \max\big(m,\ \max_{j \in \text{block}} s_j\big),\\
\ell' &= e^{m - m'}\,\ell + \sum_{j \in \text{block}} e^{s_j - m'},\\
\tilde o' &= e^{m - m'}\,\tilde o + \sum_{j \in \text{block}} e^{s_j - m'}\, v_j .
\end{aligned}
$$

换句话说：到目前为止累积的所有东西都是相对于旧最大值 $m$ 算的；乘以 $e^{m-m'}$ 就把它们转换成相对于新最大值 $m'$（因为 $e^{s-m}\cdot e^{m-m'} = e^{s-m'}$）。处理完最后一块之后，$o = \tilde o / \ell$ 就**恰好**是 softmax 加权平均——不是近似。（对块数做归纳：如果处理 $k$ 块之后 $\ell$ 和 $\tilde o$ 等于相对于 $m$ 的完整求和，那么这个更新会让它们在处理 $k+1$ 块之后等于相对于 $m'$ 的求和。）

同样的代数还允许我们**合并两个部分结果**，它们分别在不相交的 key 集合上独立算出：给定 $(m_1, \ell_1, \tilde o_1)$ 和 $(m_2, \ell_2, \tilde o_2)$，令 $m = \max(m_1, m_2)$、$\ell = e^{m_1-m}\ell_1 + e^{m_2-m}\ell_2$、$\tilde o = e^{m_1-m}\tilde o_1 + e^{m_2-m}\tilde o_2$。这个合并算子是 Flash-Decoding 和下文 cascade attention 的基础。

### FlashAttention：分块 {#flashattention-tiling}

FlashAttention 把 $Q$ 切成大小为 $B_r$ 的行块，把 $K, V$ 切成大小为 $B_c$ 的块，大小选得使这些分块能放进一个 SM 的共享内存。对每个 $Q$ 块（跨 SM 并行）：

1. 把 $Q$ 分块载入 SRAM；为它的各行初始化 $m = -\infty$、$\ell = 0$、$\tilde O = 0$。
2. 对每个 $K, V$ 块：把分块载入 SRAM，**在 SRAM/寄存器里**计算 $B_r \times B_c$ 的分数分块 $S_{ij} = Q_i K_j^\top/\sqrt d$，并应用在线 softmax 更新。
3. 把 $O_i = \tilde O / \ell$（以及 $m + \log \ell$，即 log-sum-exp，反向传播需要它）写入 HBM。

$N \times N$ 的矩阵从不接触 HBM。$Q$ 和 $O$ 的 HBM 流量只读/写一次；$K$ 和 $V$ 对每个 $Q$ 块流式读一遍（这些重复读取往往由 L2 缓存承担）。FlashAttention 论文证明，对于大小为 $M$ 的 SRAM，HBM 访问量是 $\Theta(N^2 d^2 / M)$，而标准实现是 $\Theta(Nd + N^2)$，额外内存是 $O(N)$ 而不是 $O(N^2)$。对于因果 attention，完全位于对角线上方的分块会被跳过，工作量减半。

### FlashAttention-2 和 -3 {#flashattention-2-and-3}

- **FlashAttention-2** 保持算法不变，重新组织工作：非矩阵乘操作更少（累加器只在最后缩放一次，而不是每个块都缩放）、除了 batch 和头之外还在**序列**维度上并行（这样长序列、小 batch 也能占满 GPU）、并在线程块内的 warp 之间做更好的工作划分，避免经由共享内存的通信。
- **FlashAttention-3** 面向 Hopper GPU（H100）。Hopper 可以异步地在 HBM 与共享内存之间拷贝分块（TMA 单元），并异步地运行张量核心矩阵乘（WGMMA 指令）。FA-3 使用 **warp 特化（warp specialization）**（一部分 warp 只负责加载数据，另一部分只负责计算），并让一个分块的 softmax 与另一个分块的矩阵乘重叠，使缓慢的指数运算藏在张量核心的工作后面。它还加入了 FP8 attention，用分块量化和一种“非相干处理（incoherent processing）”技巧（随机正交变换）来降低离群值造成的量化误差。论文报告最高可达 H100 峰值 BF16 吞吐的约 75%。

### 解码 attention 与 Flash-Decoding {#decode-attention-and-flash-decoding}

解码时，每个序列每个头只贡献**一个** query，去关注它的 $L$ 个已缓存的 key。FlashAttention 在 (batch × 头 × query 块) 上并行。batch 为 1、8 个 KV 头、一个 query 块时，这只有寥寥几个线程块——在有 132 个 SM 的 H100 上，大部分 GPU 闲着，每个头只有一个 SM 在遍历很长的 KV cache。

**Flash-Decoding** 在 **key** 维度上增加了并行性：把 KV cache 切成 $s$ 个块，让不同的线程块并行地为每个块计算部分结果 $(m_c, \ell_c, \tilde o_c)$，然后在一个小的归约 kernel 里用上面的 log-sum-exp 合并将它们合并。取 $s = 16$ 时，同一个问题会启动 16 倍的线程块，在长上下文下最多能快几倍。由于解码 attention 本来就是显存受限的（3.1 章），目标是让足够多的 SM 发出加载请求以打满 HBM 带宽。

GQA 在这里也有帮助：同一组里的所有查询头可以一起对共享的那个 KV 头处理，所以每个加载进来的 K/V 分块被复用 $g$ 次——解码 attention 的算术强度从 ≈ 1 上升到 ≈ $g$ FLOP/byte。

### 分页与不规则布局：FlashInfer {#paged-and-ragged-layouts-flashinfer}

真实的引擎里没有整齐的 `[batch, seq, heads, d]` 张量。在连续批处理（3.3 章）下，每一步包含不同长度的序列，有些在预填充（很多 query），有些在解码（一个 query），而在 PagedAttention 下它们的 K/V 分散在经块表找到的块里。

- **不规则（ragged）张量**：把所有序列的 token 不加 padding 地拼接起来，并传入一个偏移数组（`indptr`），标记每个序列从哪里开始。
- **分页 KV**：传入块表（每个序列的页索引）和最后一页的长度；kernel 从这些页里收集 K/V 分块。

**FlashInfer** 是为这个世界打造的 kernel 库。它把分页和不规则 KV 统一表示为块稀疏矩阵，针对每种 attention 变体（头维度、掩码、logits 软截断、滑动窗口）即时（JIT）生成专用 kernel，并在每一步之前在 CPU 上规划一个负载均衡的工作划分，让长短序列均匀分布在各个 SM 上。它还实现了 **cascade attention**：当许多请求共享一个前缀时，对共享前缀的 attention 只为它们所有的 query 计算一次，再用同样的合并算子与每个请求自己后缀的 attention 合并。vLLM 和 SGLang 把 FlashInfer 用作其 attention 后端之一；DeepSeek 的 **FlashMLA** 则在 MLA 解码中扮演同样的角色。

### Kernel 启动开销与 CUDA graph {#kernel-launch-overhead-and-cuda-graphs}

从 CPU 启动一个 kernel 的开销大约是几微秒。一个有 32–80 层的 Transformer，一个解码步会启动数百到数千个 kernel（norm、投影、RoPE、attention、激活、all-reduce……）。对于小模型或小 batch，每个 kernel 只运行几微秒，CPU 启动 kernel 的速度跟不上，GPU 在 kernel 之间空转——这一步就变成了**启动受限（launch-bound）**。

**CUDA graph** 解决了这个问题：把整个 kernel 启动序列录制一次，之后用一次启动回放。限制是 graph 是固定的：每次回放时张量形状和内存地址必须相同。因此引擎会为一组批大小（例如 1、2、4、8、16……直到某个最大值）各捕获一个 graph，把每一步的输入拷贝到静态输入缓冲区，并把 batch 填充到最近的已捕获大小。形状可变的预填充通常不用 graph 运行（它足够长，启动开销无关紧要）。`torch.compile` 的 `mode="reduce-overhead"` 会自动使用 CUDA graph。

### Kernel 融合 {#kernel-fusion}

逐元素操作（加法、乘法、激活、归一化）每读一个元素只做约 1 次 FLOP，所以如果是独立的 kernel，每一个都是对激活张量的一次完整 HBM 往返。把一串这样的操作**融合**成一个 kernel，只需读一次输入、写一次输出。LLM 推理中的标准融合有：

- 残差加法 + RMSNorm 放在一个 kernel 里；
- SwiGLU：$\text{SiLU}(x W_1) \odot (x W_3)$ 计算为一个更宽的矩阵乘加一个融合的激活-乘法；
- QKV 投影作为一个矩阵乘，RoPE 在 attention 内部或某个融合 kernel 中应用；
- “epilogue 融合”：在矩阵乘 kernel 里，在写出结果之前应用 bias、激活或量化；
- 融合的 MoE kernel：在一个分组 GEMM 里完成路由、收集 token 以及运行所有专家的矩阵乘。

## 代码 {#code}

NumPy 参考实现：朴素 attention、带在线 softmax 的分块 FlashAttention（因果），以及带 split-KV 和 log-sum-exp 合并的 Flash-Decoding。三者必须在浮点精度内一致。

```python
import numpy as np
rng = np.random.default_rng(0)

def naive_attention(Q, K, V, causal=True):
    N, d = Q.shape
    S = Q @ K.T / np.sqrt(d)                              # [N, N] 被物化
    if causal:
        S = np.where(np.tril(np.ones((N, len(K)), bool), k=len(K) - N), S, -np.inf)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P / P.sum(axis=1, keepdims=True)) @ V

def flash_attention(Q, K, V, Br=16, Bc=16):
    """因果 attention，按 [Br x Bc] 分块处理；从不构建 N x N 矩阵。"""
    N, d = Q.shape
    O = np.zeros_like(Q)
    for i in range(0, N, Br):                             # 每个 Q 块 = 一个“线程块”
        q = Q[i:i + Br]
        m = np.full(len(q), -np.inf)                      # 滚动最大值       [Br]
        l = np.zeros(len(q))                              # 滚动求和         [Br]
        acc = np.zeros((len(q), d))                       # 未归一化的输出   [Br, d]
        for j in range(0, i + len(q), Bc):                # 跳过对角线上方的分块
            s = q @ K[j:j + Bc].T / np.sqrt(d)            # 分数分块 [Br, Bc]（在“SRAM”里）
            rows, cols = np.arange(i, i + len(q))[:, None], np.arange(j, j + s.shape[1])
            s = np.where(cols <= rows, s, -np.inf)        # 分块内部的因果掩码
            m_new = np.maximum(m, s.max(axis=1))
            p = np.exp(s - m_new[:, None])
            scale = np.exp(m - m_new)                     # 缩放旧的累加器
            l = scale * l + p.sum(axis=1)
            acc = scale[:, None] * acc + p @ V[j:j + Bc]
            m = m_new
        O[i:i + Br] = acc / l[:, None]
    return O

def flash_decoding(q, K, V, n_splits=4):
    """一个解码 query q:[d] 对 L 个已缓存的 key，KV 被切成并行处理的块。"""
    d = q.shape[0]
    parts = []
    for Kc, Vc in zip(np.array_split(K, n_splits), np.array_split(V, n_splits)):
        s = Kc @ q / np.sqrt(d)                           # 每个块：独立的线程块
        m = s.max(); p = np.exp(s - m)
        parts.append((m, p.sum(), p @ Vc))                # (最大值, 求和, 未归一化的输出)
    m = max(pm for pm, _, _ in parts)                     # log-sum-exp 合并（归约）
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

引擎如何在 PyTorch 中对解码使用 CUDA graph（需要 CUDA GPU；模式——静态缓冲区、预热、捕获、拷入、回放——正是 vLLM 和 SGLang 对每个批大小所做的）：

```python
import torch

if torch.cuda.is_available():
    model = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024) for _ in range(40)]).cuda().half()
    static_in = torch.zeros(8, 1024, device="cuda", dtype=torch.half)  # 批大小为 8 的 graph

    s = torch.cuda.Stream()                    # 捕获之前在旁路 stream 上预热
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            model(static_in)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_out = model(static_in)          # 40 次 kernel 启动只录制一次

    new_batch = torch.randn(8, 1024, device="cuda", dtype=torch.half)
    static_in.copy_(new_batch)                 # 把输入写进已捕获的缓冲区
    graph.replay()                             # 一次启动回放全部 40 个 kernel
    print(torch.allclose(static_out, model(new_batch), atol=1e-2))
```

## 实例演算 {#worked-example}

**单个头的朴素 vs flash。** $N = 8{,}192$ 个 token，$d = 128$，BF16（$b = 2$）。

- FLOPs：$4N^2d = 4 \times 8192^2 \times 128 \approx 3.4\times10^{10}$。
- 朴素实现的中间流量：$S$ 和 $P$ 各有 $N^2 = 6.7\times10^7$ 个元素 = 128 MiB。两者各写一次读一次：$\approx 537$ MB。强度 $\approx 3.4\times10^{10} / 5.4\times10^8 \approx 64$ FLOP/byte → 在 H100 上是显存受限（拐点 ≈ 295）。如果有 32 个头、batch 为 8，仅分数矩阵就需要 $256 \times 128\ \text{MiB} = 32$ GiB 的暂存内存。
- FlashAttention：必须的流量是读 $Q, K, V$ 并写 $O$：每个头 $4 \times 8192 \times 128 \times 2$ B = 8 MiB，并且不需要 $N^2$ 的暂存内存。$K$ 和 $V$ 对每个 $Q$ 块会被重新流式读取，但这些读取比物化 $S$ 和 $P$ 便宜得多，而且往往由 L2 承担。kernel 变成计算受限，这就是为什么 FA-3 能达到张量核心峰值的很大一部分。

**Flash-Decoding 的占用率。** Batch 1、Llama-3-8B（32 个查询头、8 个 KV 头）、上下文 32K、H100 有 132 个 SM。让每个 KV 头的查询组由一个线程块处理，就得到 8 个块：约 6% 的 SM 在忙。把每个头的 32K 个 key 切成 16 个 2K 的块，得到 $8 \times 16 = 128$ 个块——几乎每个 SM 一个——所以所有 SM 并发地从 HBM 流式读取 KV。合并 kernel 只处理 $8 \times 16$ 个很小的部分结果。

**启动开销。** 假设一个 32 层模型的解码步每层启动约 10 个 kernel，外加少量额外的：约 330 次启动。每次约 4 µs，就是约 1.3 ms 的 CPU 启动时间，而 H100 上 8B BF16 模型带宽受限的一步约为 4.8 ms（3.1 章）——这是很大的开销，对 1B 模型更是主要成本。CUDA graph 回放把它降到一次启动。（具体的单次启动开销取决于 CPU、驱动和框架；重点在数量级。）

## 实践中的做法（2025–2026） {#in-practice-20252026}

- FlashAttention-2/3 和 FlashInfer 是开源引擎中默认的 attention 后端；Blackwell GPU 有自己的 kernel（例如 FlashInfer 和 TensorRT-LLM 提供 Blackwell 专用的 attention，FA-3 风格的思想也得以延续）。
- MLA 模型使用 **FlashMLA** 这样的专用解码 kernel，因为潜向量 KV 的布局与 GQA 不同。
- 所有生产引擎都对解码使用 CUDA graph，并在启动时捕获一系列批大小（这是启动要几十秒的部分原因）；vLLM V1 还使用 `torch.compile` 来融合逐元素操作。
- 长上下文解码会自动使用 split-KV：kernel 根据批大小、头数和上下文长度选择拆分数。
- Triton 和 CUTLASS/CuTe（以及更新的 Python DSL）这样的 kernel DSL，是编写这些 kernel 的主要方式；GPU MODE 讲座是个不错的入门。

## 常见误区 {#common-pitfalls-misconceptions}

- **“FlashAttention 是一种近似。”** 它是精确的（除去浮点重排序）。稀疏/线性 attention 方法才是近似；FlashAttention 不是。
- **“FlashAttention 减少了 FLOPs。”** 它执行相同的（反向传播中因重算而略多的）FLOPs；它减少的是 **HBM 流量和内存**。
- **把为预填充调优的 kernel 用于解码。** 没有 split-KV，长上下文、小 batch 的解码会让大部分 SM 闲置。
- **在 CUDA graph 下改变形状。** 用重新分配过（地址改变）的张量去回放 graph，会悄悄读到过期内存。一定要拷贝进已捕获的静态缓冲区。
- **忘记 split-KV 会改变浮点求和顺序。** 不同批大小和拆分数下，结果可能在最后几位不同——这与确定性有关（3.8 章）。

## 本章要点 {#key-takeaways}

- 朴素 attention 把 $N \times N$ 的中间结果写入 HBM：算术强度 ≈ $d/b$（$d=128$、BF16 时约 64 FLOP/byte）→ 显存受限，且内存为 $O(N^2)$。
- 在线 softmax 维护滚动的最大值、求和与未归一化的输出，按 $e^{m - m'}$ 缩放；结果是精确的，部分结果可以用 log-sum-exp 合并。
- FlashAttention 让 Q/K/V 分块流过 SRAM，分数从不到达 HBM；FA-2 改进并行度和工作划分；FA-3 利用 Hopper 的异步特性并加入 FP8。
- 解码 attention 需要在 key 上并行（Flash-Decoding），并支持分页/不规则布局（FlashInfer）。
- CUDA graph 消除解码中的 kernel 启动开销；融合消除逐元素操作之间的 HBM 往返。

## 自测 {#check-your-understanding}

??? question "在线 softmax 中，滚动最大值从 3 增加到 5。旧的 ℓ 和 õ 要乘以什么因子，为什么？"
    乘以 $e^{3-5} = e^{-2}$。旧的项存储为 $e^{s - 3}$；要把它们表达成相对于新最大值的形式，必须变成 $e^{s-5} = e^{s-3} e^{-2}$。

??? question "为什么 FlashAttention 的 HBM 流量不依赖 N²，而朴素 attention 的依赖？"
    $N \times N$ 的分数和概率分块只存在于 SRAM/寄存器中，并立刻被消费；只有 $Q, K, V$（读）和 $O$（写）接触 HBM，它们各为 $O(Nd)$（外加 K/V 分块的重复流式读取，这取决于 SRAM 的大小）。

??? question "为什么在批大小为 256 时不需要 Flash-Decoding？"
    256 个序列 × 8 个 KV 头 = 2,048 个独立的（序列，头）单元，并行度已经远多于 SM 数。再拆分 KV 只会增加合并开销。

??? question "为什么预填充不能对每个请求都使用同一个 CUDA graph？"
    预填充的形状取决于 prompt 长度，每个请求都不同；graph 要求形状和地址固定。预填充 kernel 也足够长，启动开销可以忽略。

??? question "是什么让 cascade attention 在数学上成为可能？"
    部分 softmax 结果的合并算子：在（共享前缀 ∪ 自身后缀）上的 attention，等于前缀上的 attention 与后缀上的 attention 做 log-sum-exp 合并，所以前缀部分可以为许多 query 只算一次。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 | 分块 + 在线 softmax 算法及其 IO 复杂度分析。 |
| :material-book-open-variant: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao, 2023 | 序列维度的并行以及更好的 warp 级工作划分。 |
| :material-new-box: [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Shah et al., 2024 | Hopper 的 warp 特化、softmax/矩阵乘重叠、FP8 attention。 |
| :material-book-open-variant: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — Milakov & Gimelshein, 2018 | FlashAttention 所建立在其上的在线 softmax 原始推导。 |
| :material-book-open-variant: [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) — Dao et al., 2023 (blog) | split-KV 解码 attention，让长上下文、小 batch 的解码用满整个 GPU。 |
| :material-book-open-variant: [FlashInfer: Efficient and Customizable Attention Engine for LLM Serving](https://arxiv.org/abs/2501.01005) — Ye et al., MLSys 2025 | 针对分页/不规则 KV 的 JIT attention kernel、负载均衡调度、cascade attention；被 vLLM 和 SGLang 使用。 |
| :material-book-open-variant: [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | DeepSeek 针对多头潜在注意力的解码 kernel。 |
| :material-book-open-variant: [GPU MODE lectures](https://github.com/gpu-mode/lectures) | 关于 CUDA、Triton 以及如何编写此类 kernel 的社区讲座。 |
