# 1.4 现代 Transformer 块

!!! abstract "本章你将学到"
    - 把 2019 年的 GPT 块变成 2025 年 Llama/Qwen/DeepSeek 块的五项升级：**RMSNorm**、**SwiGLU**、**RoPE**、**GQA/MLA**、**QK-norm**（外加共享嵌入）。
    - 为什么 SwiGLU 的隐藏维度约为 $\tfrac{8}{3}d$，以及 Llama 如何得到 11,008。
    - 从二维旋转出发推导旋转位置编码，并用数值方法验证它的相对位置性质。
    - MHA → MQA → GQA → MLA 如何一步步缩小 KV cache，附带显存计算的实例演算。
    - 一个完整、可运行的 Llama 风格块的 PyTorch 实现。

## 为什么重要 {#why-this-matters-the-problem}

你在 [1.3](03-transformer.md) 中搭的 GPT 是可以工作的，但它已经不是 2025 年任何人会训练的东西了。2019 到 2024 年间，社区收敛到了一组小改动，每一项都买到了具体的东西：每 FLOP 稍好的质量、大规模下更稳定的训练、对位置和长上下文更好的处理，以及最重要的，一个小得多的 **KV cache**，也就是决定模型服务成本的那块显存。其结果常被称为"Llama 块"，尽管 Qwen、DeepSeek、Mistral、Gemma 和 gpt-oss 用的都是它的变体。理解每个改动及其原因，能让你读到任何模型配置时都知道它意味着什么。

## 直觉 {#intuition}

| GPT-2 块（2019） | 现代块（2023–2025） | 它带来了什么 |
|---|---|---|
| LayerNorm | **RMSNorm** | 同样的稳定性，更少的计算，无需减均值 |
| GELU MLP，$d_{ff} = 4d$ | **SwiGLU** MLP，$d_{ff} \approx \tfrac{8}{3}d$ | 在相同参数量和 FLOPs 下损失更低 |
| 学习式绝对位置 | **RoPE**（旋转 q 和 k） | 相对位置，没有长度受限的表，可扩展上下文 |
| 多头注意力 | **GQA** 或 **MLA** | KV cache 缩小 4–60 倍，解码更快 |
| 无 | **QK-norm**（可选） | 防止注意力 logits 爆炸 |
| 输入/输出嵌入分开 | 共享或不共享 | 共享可节省 $Vd$ 个参数（主要在小模型上） |

```text
x ─► RMSNorm ─► Attention（GQA 或 MLA，RoPE，可选 QK-norm）─► + ─►
     RMSNorm ─► FFN（SwiGLU 稠密，或 MoE：路由器 → top-k 专家 + 共享专家）─► + ─► 下一个块
```

FFN 的 MoE 变体见 [1.5](05-mixture-of-experts.md) 章。

## 工作原理 {#how-it-works}

### RMSNorm {#rmsnorm}

LayerNorm（[1.3](03-transformer.md)）会减去均值、除以标准差，再应用缩放和平移。Zhang 与 Sennrich（2019）发现重新中心化是不必要的；只做重新缩放就能带来稳定性。**RMSNorm** 把每个向量除以它的均方根：

$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}, \qquad \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon},
$$

带一个学习得到的逐特征缩放 $\gamma \in \mathbb{R}^d$，没有偏置。用话说：把向量缩小或拉伸，使它的典型元素量级约为 1，然后让模型重新给各特征加权。它省去了一次归约（均值）和一个参数向量；实践中，即使模型以 BF16 运行，它也用 FP32 计算，因为平方和可能溢出或损失精度。

### SwiGLU 前馈网络 {#swiglu-feed-forward}

**门控线性单元**（gated linear unit，GLU）把两个线性投影逐元素相乘，其中一个经过一个充当软门的非线性函数。Shazeer（2020）测试了多种变体，发现使用 **Swish**（也叫 **SiLU**）激活的那个效果最好，$\mathrm{Swish}(z) = z \cdot \sigma(z)$，其中 $\sigma$ 是 logistic sigmoid：

$$
\mathrm{FFN}_{\text{SwiGLU}}(x) = \big(\mathrm{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}}\big)\, W_{\text{down}},
$$

其中 $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$，$W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$。用话说：一个投影决定每个隐藏特征放行*多少*，另一个提供特征本身，两者的乘积再被投影回来。它为什么效果更好，理论上并没有完全解释清楚（Shazeer 的论文著名地把原因归结为"神的恩典"）；经验上，它在相同算力下始终更好。

**为什么 $d_{ff} \approx \tfrac{8}{3}d$。** 经典的 GELU FFN 取 $d_{ff} = 4d$，有两个矩阵，共 $2 \times d \times 4d = 8d^2$ 个参数。SwiGLU 有三个矩阵，共 $3 d\, d_{ff}$ 个参数。为了让参数量（和 FLOPs）保持相同，令 $3 d\, d_{ff} = 8 d^2$，得 $d_{ff} = \tfrac{8}{3} d$。对 Llama 的 $d = 4096$：$\tfrac{8}{3} \times 4096 = 10{,}922.7$，Llama 为了硬件效率向上取整到 256 的倍数：$43 \times 256 = 11{,}008$。后来的模型对 $d_{ff}$ 的选择更自由（Llama 3 8B 用 14,336 = $3.5d$）。

### RoPE：旋转位置编码 {#rope-rotary-position-embeddings}

我们希望注意力分数依赖于 token *在哪里*，最好只依赖它们的**相对**距离 $n - m$（像"关注前一个 token"这样的模式，应当在序列的任何地方都能工作）。Su 等（2021）通过旋转查询和键来实现这一点。

**先看二维。** 设 $R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}$ 为角度 $\alpha$ 的二维旋转。把位置 $m$ 处 token 的查询通过 $R(m\theta)$，位置 $n$ 处的键通过 $R(n\theta)$，其中 $\theta$ 是某个频率。分数变为

$$
\big(R(m\theta) q\big)^\top \big(R(n\theta) k\big) = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\big((n - m)\theta\big)\, k,
$$

因为旋转矩阵是正交的（$R(\alpha)^\top = R(-\alpha)$），而且旋转的复合是角度相加。用话说：按绝对位置旋转两个向量之后，分数只依赖它们的偏移 $n - m$。旋转还保持向量长度不变，所以不会改变分数的尺度。

**多维情形。** 把 $d_h$ 维的头向量拆成 $d_h/2$ 对，第 $i$ 对旋转角度 $m\theta_i$，每对使用不同的频率：

$$
\theta_i = b^{-2i/d_h}, \qquad i = 0, 1, \dots, \tfrac{d_h}{2} - 1,
$$

其中 $b$ 是 **RoPE 基数**（原始为 10,000）。前面的对旋转得快（每个 token 角度变化约 1 弧度），编码精细的局部位置；后面的对旋转得非常慢（波长 $2\pi/\theta_i$ 最长约为 $2\pi b$ 个 token），编码粗粒度的长程位置，就像时钟的各根指针。总分数是各对之和，每一对都只依赖 $n - m$，所以相对位置性质对整个向量成立。

RoPE 只作用于 $q$ 和 $k$（不作用于 $v$），在每个注意力层内部，并且不增加参数。因为频率是基数的简单函数，它也是重新缩放频率的**上下文扩展**方法的切入点（[1.6](06-long-context-and-efficient-attention.md)）。

!!! note "两种布局"
    原论文把相邻维度 $(0,1), (2,3), \dots$ 配成对。Llama/Hugging Face 的约定是把维度 $i$ 与 $i + d_h/2$ 配对（"rotate half"）。两者都合法，但用一种布局训练出的权重必须以同样的布局使用。

### 从 MHA 到 MQA、GQA 和 MLA：缩小 KV cache {#from-mha-to-mqa-gqa-and-mla-shrinking-the-kv-cache}

生成时，每个新 token 都必须关注所有之前的 token。每一步都重新计算它们的键和值太浪费，所以推理引擎把它们存起来：这就是 **KV cache**（[第三部分](../part3/index.md)）。它每个 token 的大小为

$$
\text{KV bytes per token} = 2 \times L \times n_{kv} \times d_h \times \text{bytes per element},
$$

其中 2 对应 K 和 V，$L$ 是层数，$n_{kv}$ 是键/值头数，$d_h$ 是头大小。解码受限于显存带宽：每一步都要读取整个缓存。缓存越小，意味着每块 GPU 能服务更多并发用户、更长的上下文、更快的 token 速度。

- **MHA（multi-head attention，多头注意力）：** $n_{kv} = h$，每个查询头都有自己的 K 和 V。
- **MQA（multi-query attention，多查询注意力，Shazeer 2019）：** $n_{kv} = 1$，所有查询头共享同一个 K/V 头。缓存缩小为 $1/h$，但有一些质量损失和训练不稳定。
- **GQA（grouped-query attention，分组查询注意力，Ainslie 等 2023）：** $1 < n_{kv} < h$。查询头被分成 $n_{kv}$ 组，每组 $h / n_{kv}$ 个；每组共享一个 K/V 头。取 $n_{kv} = 8$ 时，质量接近 MHA，缓存缩小为 $h/8$ 分之一。实现方式：计算 $n_{kv}$ 个 K/V 头，并在注意力之前把每个重复 $h/n_{kv}$ 次。
- **MLA（multi-head latent attention，多头潜在注意力，DeepSeek-V2 2024）：** 不存储每个头的键和值，而是每个 token 存一个压缩的**潜在向量**，并从它重建 K 和 V。

**MLA 的更多细节。** 对于隐藏状态为 $x$ 的 token：

1. 向下投影到一个潜在向量：$c^{KV} = W^{DKV} x \in \mathbb{R}^{d_c}$，其中 $d_c$ 远小于 $h \cdot d_h$（DeepSeek-V3：$d_c = 512$，而 $h \cdot d_h = 128 \times 128 = 16{,}384$）。
2. 向上投影到每个头的键和值：对每个头 $m$，$k^C_m = W^{UK}_m c^{KV}$，$v_m = W^{UV}_m c^{KV}$。
3. 只缓存 $c^{KV}$。推理时，上投影可以被**吸收**：$q_m^\top k^C_m = q_m^\top W^{UK}_m c^{KV} = (W^{UK\top}_m q_m)^\top c^{KV}$，所以查询只需一次性映射进潜在空间，并直接关注缓存的潜在向量；类似地，$W^{UV}$ 折进输出投影。
4. **解耦 RoPE。** 如果在 $W^{UK}$ 和查询之间夹着一个与位置有关的旋转，吸收就会失效，所以 MLA 保留了一个单独的小的键分量 $k^R$（DeepSeek-V3：64 维）来携带 RoPE，并被所有头共享；它与 $c^{KV}$ 一起缓存。每个头的查询同样被拆成不带 RoPE 的部分和带 RoPE 的部分。

于是缓存是每个 token 每层 $(d_c + d_R)$ 个元素，其中 $d_R$ 是 RoPE 键的大小，与头数无关。DeepSeek-V2 报告表明，MLA 的质量与 MHA 持平甚至更好，而 MQA/GQA 则是用质量换显存。

### QK-norm {#qk-norm}

在大规模下，注意力 logits $q \cdot k$ 在训练中可能无限增长，使 softmax 饱和，并导致损失尖峰或发散。**QK-norm** 在点积之前（也在 RoPE 之前）对每个头的查询和键向量施加一个归一化（RMSNorm 或 LayerNorm，带学习得到的缩放）。这限制了分数的量级。它曾被用来稳定 Dehghani 等（2023）的 22B 参数 ViT，现在 OLMo 2、Qwen3 和 Gemma 3 都在使用。[1.8](08-training-at-scale.md) 章会讨论其他稳定训练的办法。

### 共享嵌入 {#tied-embeddings}

输入嵌入 $E \in \mathbb{R}^{V \times d}$ 和输出投影 $W_U \in \mathbb{R}^{d \times V}$ 都在把 token 与 $d$ 维向量联系起来。**共享**（tying）令 $W_U = E^\top$（Press 与 Wolf，2016），节省 $Vd$ 个参数。对小模型来说这是很大的一部分：当 $V = 151{,}936$、$d = 1024$ 时，$Vd \approx 156\text{M}$，占 0.6B 模型的很大份额。对大模型，节省的只是很小一部分，而不共享的矩阵能提供稍多的灵活性，所以大多数大模型（Llama 3、DeepSeek-V3、Qwen3 的大尺寸版本）不共享，而小模型（Gemma、Qwen3 0.6B–4B）共享。

## 代码 {#code}

完整的 Llama/Qwen 风格解码器：RMSNorm、RoPE、带可选 QK-norm 的 GQA、SwiGLU、Pre-norm 残差、无偏置。它使用 PyTorch 融合的 `scaled_dot_product_attention`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):                          # 在最后一维上归一化
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight

def rope_cache(T, dh, base=10000.0):
    # theta_i = base^(-2i/dh), i = 0..dh/2-1 ; angle[m, i] = m * theta_i
    theta = base ** (-torch.arange(0, dh, 2).float() / dh)       # [dh/2]
    ang = torch.outer(torch.arange(T).float(), theta)            # [T, dh/2]
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):
    # x: [B, heads, T, dh]；"对半拆分"约定：维度 i 与维度 i + dh/2 配对
    x1, x2 = x.chunk(2, dim=-1)                                  # 各为 [B, heads, T, dh/2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class GQAttention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, qk_norm=True):
        super().__init__()
        self.h, self.kv, self.dh = n_heads, n_kv_heads, d // n_heads
        self.wq = nn.Linear(d, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.dh, bias=False)  # K/V 头更少
        self.wv = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d, bias=False)
        self.q_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()

    def forward(self, x, cos, sin):                               # x: [B, T, d]
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)   # [B, h, T, dh]
        k = self.wk(x).view(B, T, self.kv, self.dh).transpose(1, 2)  # [B, kv, T, dh]
        v = self.wv(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)                     # QK-norm（逐头）
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # 通过旋转引入位置
        rep = self.h // self.kv                                   # 每个 KV 头对应的查询头数
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, h, T, dh]
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))
```

```python
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(d), RMSNorm(d)
        self.attn = GQAttention(d, n_heads, n_kv_heads)
        self.ffn = SwiGLU(d, d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))

class LlamaStyleLM(nn.Module):
    def __init__(self, V, d=256, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=688,
                 max_len=1024, rope_base=10000.0, tie=False):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, n_kv_heads, d_ff) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if tie:
            self.head.weight = self.emb.weight          # 共享嵌入
        cos, sin = rope_cache(max_len, d // n_heads, rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):                             # idx: [B, T]
        T = idx.shape[1]
        x = self.emb(idx)                               # 没有位置嵌入：位置由 RoPE 负责
        for blk in self.blocks:
            x = blk(x, self.cos[:T], self.sin[:T])
        return self.head(self.norm(x))                  # [B, T, V]

model = LlamaStyleLM(V=1000)
logits = model(torch.randint(0, 1000, (2, 16)))
print(logits.shape, sum(p.numel() for p in model.parameters()))  # [2, 16, 1000], 3283456
```

注意 $d_{ff} = 688 \approx \tfrac{8}{3} \times 256$，取整到 16 的倍数。把这个模型换进 [1.3](03-transformer.md) 的训练循环（它只返回 logits，所以交叉熵要在外面计算），就可以与 GPT-2 风格的块做比较。

**用数值方法检验相对位置性质。** 使用上面的 `rope_cache` 和 `apply_rope`：

```python
torch.manual_seed(0)
dh = 8
cos, sin = rope_cache(64, dh)
q, k = torch.randn(1, 1, 1, dh), torch.randn(1, 1, 1, dh)

def score(m, n):  # 位置 m 处的查询，位置 n 处的键
    qm = apply_rope(q, cos[m:m+1], sin[m:m+1])
    kn = apply_rope(k, cos[n:n+1], sin[n:n+1])
    return (qm * kn).sum().item()

print(score(5, 2), score(40, 37))   # 偏移相同（3）  -> 分数相同
print(score(10, 10), score(0, 0))   # 偏移相同（0）  -> 分数相同
```

两对打印出的值相等（除去浮点舍入误差）：分数只依赖偏移。

## 实例演算 {#worked-example}

**70B 级模型的 KV cache。** Llama 3 70B 有 $L = 80$ 层，$h = 64$ 个大小为 $d_h = 128$ 的查询头，并使用 GQA，$n_{kv} = 8$。使用 BF16（每个元素 2 字节）：

| 变体 | $n_{kv}$ | 每 token 字节数：$2 \cdot 80 \cdot n_{kv} \cdot 128 \cdot 2$ | 128K token 时（131,072） |
|---|---|---|---|
| MHA | 64 | 2,621,440 B = 2.5 MiB | 320 GiB |
| **GQA（实际采用）** | 8 | 327,680 B = 320 KiB | 40 GiB |
| MQA | 1 | 40,960 B = 40 KiB | 5 GiB |

用 MHA 的话，一段 128K token 的对话就需要 320 GiB 的缓存，比四块 80 GB 的 GPU 能装下的还多，这还没算 140 GB 的 BF16 权重。GQA-8 把它降到 40 GiB。

**DeepSeek-V3 的 MLA。** $L = 61$，潜在向量 $d_c = 512$，解耦 RoPE 键 $d_R = 64$：

$$
(512 + 64) \times 61 \times 2 \text{ bytes} = 70{,}272 \text{ B} \approx 68.6 \text{ KiB per token},
$$

在 128K token 时约 8.6 GiB。如果同一个模型使用完整的 MHA 和它的 128 个头（键为 128 + 64 维，值为 128 维），每个 token 将需要 $128 \times (192 + 128) \times 61 \times 2 \approx 4.77$ MiB，约大 71 倍。MLA 的缓存比上面（更小的）70B 模型的 GQA-8 缓存还要小。

**SwiGLU 尺寸核对。** 对 $d = 4096$，隐藏维度为 $4d$ 的 GELU FFN 每层有 $8d^2 = 134.2\text{M}$ 个参数；Llama 的 SwiGLU 取 $d_{ff} = 11{,}008$ 时有 $3 \times 4096 \times 11{,}008 = 135.3\text{M}$，相差在 1% 以内。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **所有人都在用** Pre-RMSNorm、SwiGLU（或相近变体）、RoPE，以及主要线性层中不带偏置。这套组合由 LLaMA（2023）推广开来。
- **GQA** 是稠密模型的默认选择：Llama 3（所有尺寸都是 8 个 KV 头）、Qwen3（例如 Qwen3-235B-A22B 为 64 个查询头 / 4 个 KV 头）、Gemma 3、gpt-oss（64 个查询头 / 8 个 KV 头）。**MLA** 被 DeepSeek-V2/V3/R1 和 Kimi K2 使用。
- **RoPE 基数**为支持长上下文而不断增大：10,000（原始版本，Llama 2）→ 500,000（Llama 3）→ 1,000,000（Qwen3、Gemma 3 的全局层）。
- **QK-norm** 出现在 OLMo 2、Qwen3 和 Gemma 3 中。Kimi K2 使用 MLA，在其中 QK-norm 很别扭（吸收式推理时键从不被具体化），所以它改为在训练中裁剪查询/键的权重（MuonClip，[1.8](08-training-at-scale.md)）。
- **有些层跳过 RoPE。** Llama 4 把不带位置编码的层（"NoPE"）与 RoPE 层交错排列，以帮助长度泛化。

## 常见误区 {#common-pitfalls-misconceptions}

- **把 RoPE 用在值上**，或用在 QK 点积之后。它只旋转 q 和 k，并且在点积之前。
- **混用 RoPE 布局**（交错 vs 对半拆分）来移植权重：一切都能运行，输出却是乱码。
- **以为 GQA 能大幅减少 FLOPs。** 它主要缩小的是 K/V 投影和缓存；分数计算仍然对每个查询头进行。好处是推理时的显存和带宽。
- **以为 MLA 只是低秩 K/V。** 关键细节是推理时的权重吸收和解耦 RoPE；没有它们，你每一步都得解压缓存，也就失去了大部分收益。
- **从 GELU 配置里照搬 SwiGLU 的隐藏维度。** 把 $d_{ff} = 4d$ 抄进 SwiGLU 层，会悄悄地让 FFN 参数增加 50%。

## 本章要点 {#key-takeaways}

- RMSNorm 去掉了 LayerNorm 的均值中心化；SwiGLU 对 FFN 加门控，并用 $d_{ff} \approx \tfrac{8}{3}d$ 保持参数量不变。
- RoPE 按与位置相关的角度旋转查询/键的各对分量，使分数只依赖相对位置；基数决定频率范围。
- KV cache 每个 token 花费 $2 L n_{kv} d_h$ 个元素；GQA 减小 $n_{kv}$，MLA 改为缓存一个小的潜在向量，两者对廉价的长上下文服务都至关重要。
- QK-norm 使注意力 logits 保持有界，稳定大规模训练。
- 共享嵌入对小模型重要，对大模型不重要。

## 自测 {#check-your-understanding}

??? question "某模型 $L = 32$，$h = 32$，$d_h = 128$，GQA 取 $n_{kv} = 8$，缓存为 FP16。每个 token 的 KV cache 是多少？32K 上下文又是多少？"
    $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ 字节 = 每 token 128 KiB。对 32,768 个 token：$128\text{ KiB} \times 32{,}768 = 4$ GiB。

??? question "用一行证明 RoPE 分数只依赖 $n - m$。"
    $(R(m\theta)q)^\top(R(n\theta)k) = q^\top R(-m\theta)R(n\theta)k = q^\top R((n-m)\theta)k$，用到 $R(\alpha)^\top = R(-\alpha)$ 和 $R(\alpha)R(\beta) = R(\alpha + \beta)$；这对每个二维对成立，而完整的分数是各对之和。

??? question "为什么增大 RoPE 基数有助于长上下文？"
    更大的 $b$ 让最慢的频率变得更慢，所以它们的旋转角度在更长的距离上仍然很小（不产生歧义）。相距很远的位置依然可以被区分，而角度不会绕回来。

??? question "某 SwiGLU 模型 $d = 5120$。什么样的 $d_{ff}$ 能匹配 $4d$ GELU FFN 的参数量？"
    $\tfrac{8}{3} \times 5120 \approx 13{,}653$；向上取整到 256 的倍数是 $54 \times 256 = 13{,}824$。

??? question "为什么 MLA 不能直接对重建出来的键应用标准 RoPE？"
    键会变成 $R_n W^{UK} c$，与位置有关的旋转夹在 $W^{UK}$ 和查询之间，所以一旦如此，$W^{UK}$ 就不能再被一次性吸收进查询投影。每一步都需要重建所有过去的键。解耦 RoPE 把位置信息放进一个单独的小键里，直接缓存它。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [RMSNorm](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019 | 不做均值中心化的 LayerNorm：更便宜，效果一样好。Llama、Qwen、DeepSeek、Gemma 的标准配置。 |
| :material-star: [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202) — Shazeer, 2020 | 门控 FFN `(Swish(xW) ⊙ xV)W₂` 在相同算力下胜过 ReLU/GELU FFN。如今到处都是默认的 FFN。 |
| :material-star: [RoFormer: Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — Su et al., 2021 | 通过旋转查询/键向量来编码位置，使注意力依赖相对距离。 |
| :material-star: [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — Shazeer, 2019 | 所有查询头共享一个 K/V 头，KV cache 缩小为头数分之一。 |
| :material-star: [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023 | 一组查询头共享一个 K/V 头：质量与显存的最佳平衡点。还展示了如何把 MHA 检查点转换为 GQA。 |
| :material-star: [DeepSeek-V2 (Multi-head Latent Attention, MLA)](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024 | 带解耦 RoPE 的低秩潜在 KV cache；以小得多的缓存达到 MHA 的质量。 |
| :material-book-open-variant: [Scaling Vision Transformers to 22B Parameters](https://arxiv.org/abs/2302.05442) — Dehghani et al., 2023 | 最早使用 **QK-norm** 来防止注意力 logits 爆炸的工作之一。 |
| :material-book-open-variant: [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) — Press & Wolf, 2016 | 支持共享输入与输出嵌入的论据。 |
| :material-book-open-variant: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023 | 把 Pre-RMSNorm + SwiGLU + RoPE 打包成开源模型的"标准块"。 |
| :material-star: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Llama Team, Meta, 2024 | 完整的稠密 405B 流水线；其架构部分是现代稠密块的精炼参考（在 [1.9](09-frontier-model-case-studies.md) 中会再次回顾）。 |
