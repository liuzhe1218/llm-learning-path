# 1.3 Transformer

!!! abstract "本章你将学到"
    - token id 如何变成向量（嵌入），以及位置如何被编码。
    - 缩放点积注意力，带显式张量形状地一步步推导，包括为什么要除以 $\sqrt{d_k}$。
    - 因果掩码、多头注意力、前馈网络、残差连接和 LayerNorm（Pre-LN 与 Post-LN）。
    - 一个约 100 行 PyTorch 的完整最小 GPT，你可以在笔记本上训练它。
    - 如何数清参数量和 FLOPs（每个训练 token 约 $6N$，每个推理 token 约 $2N$），并以 7B 级模型为例演算。

## 为什么重要 {#why-this-matters-the-problem}

[1.1](01-language-modeling.md) 章把语言建模归结为：为每个位置计算 $p_\theta(x_t \mid x_{<t})$。我们需要一个函数，它 (a) 能读入任意长度的前缀，(b) 让每个位置都能使用来自*任何*更早位置的信息，而不只是固定窗口内的，并且 (c) 并行计算所有位置，使 GPU 上的训练高效。循环网络满足 (a) 并部分满足 (b)，但它们一个接一个地处理 token，所以训练慢，并且很难把信息传递很远。**Transformer**（Vaswani 等，2017）三条全部满足。本课程中的每个模型都是它的仅解码器后代，所以本章是后面一切的基础。

## 直觉 {#intuition}

把每个 token 想象成一个手里拿着向量的工人，这个向量是它当前在上下文里对自己的"理解"。一个 Transformer 层有两个阶段：

1. **注意力（通信）。** 每个 token 广播一个*键*（"这是我包含的内容"）和一个*值*（"这是我会交出的东西"），并发出一个*查询*（"这是我在找的东西"）。每个 token 把自己的查询与每个更早 token 的键作比较，把匹配分数变成权重，然后收集它们的值的加权平均。"it" 可以回头看，从 "the cat" 中拉取信息。
2. **前馈网络（计算）。** 然后每个 token 独自思考：一个小的两层 MLP 变换它的向量。模型的大部分事实性知识，据认为就存储在这些 MLP 权重里。

堆叠几十个这样的层，向量就从"这是 token *bank*"演化为"这是金融意义上的 *bank*，是动词的宾语，下一个 token 很可能是一个名词短语"。每个位置上的最后一个向量被投影成词表上的 logits。

```text
token id ─► 嵌入 ─►┌───────────────────────── × L 层 ──────────────────────────────┐─► norm ─► 反嵌入 ─► logits
                   │ x = x + Attention(Norm(x))    （token 与更早的 token 交流）    │
                   │ x = x + MLP(Norm(x))          （每个 token 独自思考）          │
                   └───────────────────────────────────────────────────────────────┘
```

## 工作原理 {#how-it-works}

全章统一使用的记号：批大小 $B$，序列长度 $T$，模型（隐藏）宽度 $d$（也写作 `d_model`），头数 $h$，每个头的宽度 $d_h = d / h$，层数 $L$，词表大小 $V$。

### 1. 嵌入与位置 {#1-embeddings-and-positions}

输入是一个 token id 张量 `idx: [B, T]`。**嵌入矩阵** $E \in \mathbb{R}^{V \times d}$ 把每个 id 映射成其中一行：`x = E[idx]: [B, T, d]`。这就是一张可学习的查找表。

我们接下来会看到，注意力把它的输入当作一个无序集合：打乱 token 的顺序，输出也会以同样的方式被打乱。因此词序必须被显式地注入。原始 Transformer 加的是固定的正弦向量；GPT-2 加的是**学习式位置嵌入** $P \in \mathbb{R}^{T_{\max} \times d}$，即 `x = E[idx] + P[0:T]`。现代模型则用 **RoPE**（旋转位置编码，[1.4](04-modern-block.md)）来旋转查询和键，但学习式位置最简单，适合作为起点。

### 2. 缩放点积注意力 {#2-scaled-dot-product-attention}

取一个序列，`x: [T, d]`（暂时去掉批维度）。三个学习得到的矩阵 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ 和 $W_V \in \mathbb{R}^{d \times d_v}$ 对每个 token 做投影：

$$
Q = x W_Q \in \mathbb{R}^{T \times d_k}, \qquad K = x W_K \in \mathbb{R}^{T \times d_k}, \qquad V = x W_V \in \mathbb{R}^{T \times d_v}.
$$

$Q$ 的第 $i$ 行是 token $i$ 的查询 $q_i$；$k_j$ 和 $v_j$ 同理。token $i$ 的查询与 token $j$ 的键之间的相似度就是点积 $q_i \cdot k_j$。一次矩阵乘法就能算出所有的对：

$$
S = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}, \qquad S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}.
$$

$S$ 的每一行通过对 $j$ 做 softmax 变成一个概率分布，得到**注意力权重** $A = \mathrm{softmax}_{\text{row}}(S)$，每个 token 的输出是各值向量的加权平均：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big) V \in \mathbb{R}^{T \times d_v}.
$$

用话说：输出的第 $i$ 行是 $\sum_j A_{ij} v_j$，即所有值向量的混合，权重取决于 token $i$ 的查询与各 token 的键匹配得有多好。每一行的权重非负且和为 1。

**为什么要除以 $\sqrt{d_k}$？** 假设在初始化时，$q$ 和 $k$ 的各分量相互独立、均值为 0、方差为 1。那么

$$
\mathrm{Var}(q \cdot k) = \mathrm{Var}\Big(\sum_{m=1}^{d_k} q_m k_m\Big) = \sum_{m=1}^{d_k} \mathrm{Var}(q_m)\mathrm{Var}(k_m) = d_k,
$$

因为每个乘积 $q_m k_m$ 的均值为 0、方差为 $1 \cdot 1$，而独立项的方差相加。所以原始分数的标准差是 $\sqrt{d_k}$，当 $d_k = 128$ 时约为 11.3。对分布如此分散的值做 softmax，结果几乎是 one-hot：一个权重约为 1，其余约为 0，而饱和 softmax 的梯度几乎为零，学习就停滞了。除以 $\sqrt{d_k}$ 把分数的方差拉回 1，与头的大小无关。

### 3. 因果掩码 {#3-the-causal-mask}

对语言建模来说，位置 $i$ 不能看到 $j > i$ 的 token；否则预测 $x_{i+1}$ 就变得微不足道（它就在输入里）。我们在 softmax 之前对 $j > i$ 设置 $S_{ij} = -\infty$。由于 $e^{-\infty} = 0$，这些权重恰好变为 0：

```text
          键 j →   1     2     3     4
查询 i = 1        [ s11  -inf  -inf  -inf ]
查询 i = 2        [ s21  s22   -inf  -inf ]
查询 i = 3        [ s31  s32   s33   -inf ]
查询 i = 4        [ s41  s42   s43   s44  ]
```

有了掩码，对 $T$ 个 token 的一次前向传播就能同时产生 $T$ 个有效的下一个 token 预测，每个只使用它自己的前缀。这就是 Transformer 训练能并行的原因。带这种掩码的模型叫**仅解码器**（decoder-only，因果）Transformer。

### 4. 多头注意力 {#4-multi-head-attention}

每层只有一种注意力模式是有局限的：一个 token 可能需要同时关注前一个 token（句法）、句子的主语（一致关系）和一个匹配的括号（代码）。**多头注意力**并行运行 $h$ 个注意力操作，每个都有自己的投影，映射到宽度为 $d_h = d/h$ 的更小空间：

$$
\mathrm{head}_m = \mathrm{Attention}(x W_Q^{(m)}, x W_K^{(m)}, x W_V^{(m)}), \quad m = 1, \dots, h,
$$

$$
\mathrm{MHA}(x) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W_O, \qquad W_O \in \mathbb{R}^{d \times d}.
$$

实践中，所有头通过 reshape 一起计算。批量版本的形状如下：

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 | `x` | `[B, T, d]` |
| 融合投影 | `x @ W_qkv` | `[B, T, 3d]` |
| 拆分并 reshape 成多个头 | `q, k, v` | 各为 `[B, h, T, d_h]` |
| 分数 | `q @ k.transpose(-2, -1) / sqrt(d_h)` | `[B, h, T, T]` |
| 掩码 + softmax，然后 `@ v` | `out` | `[B, h, T, d_h]` |
| 合并各头 | `out.transpose(1, 2).reshape(B, T, d)` | `[B, T, d]` |
| 输出投影 | `out @ W_O` | `[B, T, d]` |

总开销与一个宽度为 $d$ 的大头大致相同，因为每个头都在 $d_h = d/h$ 的空间里工作。

### 5. 前馈网络（FFN / MLP） {#5-the-feed-forward-network-ffn-mlp}

注意力之后，每个位置独立地通过一个两层 MLP，先扩展到隐藏宽度 $d_{ff}$（经典值为 $4d$），再投影回来：

$$
\mathrm{FFN}(x) = \phi(x W_1 + b_1)\, W_2 + b_2, \qquad W_1 \in \mathbb{R}^{d \times d_{ff}},\ W_2 \in \mathbb{R}^{d_{ff} \times d},
$$

其中 $\phi$ 是非线性函数（原始论文用 ReLU，GPT-2 用 GELU，现代模型用 SwiGLU，见 [1.4](04-modern-block.md)）。当 $d_{ff} = 4d$ 时，FFN 有 $8d^2$ 个参数，是注意力块 $4d^2$ 的两倍；参数大部分都在 FFN 里。

### 6. 残差连接与残差流 {#6-residual-connections-and-the-residual-stream}

每个子层都把输出*加*到输入上：$x \leftarrow x + F(x)$。有两个好处：

- **梯度流动。** $x + F(x)$ 对 $x$ 的导数是 $I + \partial F/\partial x$；恒等项给梯度提供了从损失直通早期层的路径，所以 100 层的网络依然可训练。
- **残差流（residual stream）视角。** 把每个位置上的向量 $x$ 想象成一条宽度为 $d$ 的共享总线，从嵌入一直运行到输出。每个注意力或 MLP 子层从总线上*读取*，再把一个更新*写回*。各层通过写入后续层会读取的子空间来通信。这个视角（由机制可解释性研究推广开来）是理解为什么深度有效、为什么最后一层能读出预测的一个有用心智模型。

### 7. LayerNorm，以及 Pre-LN 与 Post-LN {#7-layernorm-and-pre-ln-vs-post-ln}

**层归一化**（LayerNorm）把每个 token 的向量在其 $d$ 个特征上重新缩放到零均值、单位方差，然后应用一个学习得到的缩放 $\gamma$ 和平移 $\beta$：

$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu = \frac{1}{d}\sum_{i=1}^{d} x_i,\quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2,
$$

其中 $\odot$ 是逐元素乘法，$\epsilon$（例如 $10^{-5}$）用来避免除以零。它让激活值在通过许多层时保持在稳定的范围内。

放在哪里很重要：

- **Post-LN**（原始 Transformer）：$x \leftarrow \mathrm{LN}(x + F(x))$。归一化位于残差路径*上*，所以每一层的输出都被重新归一化。Xiong 等（2020）表明，在初始化时，靠近输出层的梯度很大，这就是 Post-LN 需要谨慎的学习率预热、并且在深层模型中不稳定的原因。
- **Pre-LN**（GPT-2 起）：$x \leftarrow x + F(\mathrm{LN}(x))$，并在输出投影之前加一个最终的 LN。残差路径是一个干净的求和，梯度表现良好，可以用更大的学习率稳定训练。每个现代大模型都使用 Pre-norm 设计（通常配合 RMSNorm，[1.4](04-modern-block.md)）。

### 8. 输出层 {#8-the-output-layer}

在最后一个块之后，一个最终的 norm 和一个**反嵌入**（unembedding）矩阵 $W_U \in \mathbb{R}^{d \times V}$ 产生 `logits: [B, T, V]`，再对错位后的目标计算交叉熵，就得到 [1.1](01-language-modeling.md) 中的损失。有些模型把 $W_U = E^\top$ 共享以节省参数（[1.4](04-modern-block.md)）。

## 代码 {#code}

一个 GPT-2 风格的完整 GPT：学习式位置、Pre-LN、GELU MLP。第一个代码块定义模型，第二个训练它。

```python
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # W_Q, W_K, W_V 融合在一起
        self.proj = nn.Linear(d_model, d_model)      # W_O

    def forward(self, x):                            # x: [B, T, d]
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)       # 各为 [B, T, d]
        # 把 d 拆成 h 个大小为 dh 的头：[B, T, d] -> [B, h, T, dh]
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)        # [B, h, T, T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))             # 不许偷看后面
        out = F.softmax(scores, dim=-1) @ v                          # [B, h, T, dh]
        out = out.transpose(1, 2).reshape(B, T, d)                   # 拼接各头
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.up = nn.Linear(d_model, 4 * d_model)
        self.down = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn, self.mlp = CausalSelfAttention(d_model, n_heads), MLP(d_model)

    def forward(self, x):                  # Pre-LN 残差块
        x = x + self.attn(self.ln1(x))     # token 之间交换信息
        x = x + self.mlp(self.ln2(x))      # 每个 token 独立处理
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)     # 学习式绝对位置
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # 反嵌入

    def forward(self, idx, targets=None):                # idx: [B, T] token id
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)        # [B, T, d]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))                 # [B, T, V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.pos_emb.num_embeddings:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
```

在字符上训练（把两个代码块存进同一个文件；它在自己的源码上训练，所以无需下载，但任何文本文件都可以）：

```python
torch.manual_seed(0)
text = open(__file__).read() * 20                  # 换成任意文本文件
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
model = GPT(vocab_size=len(chars))
print(sum(p.numel() for p in model.parameters()), "parameters")
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
B, T = 32, 64
for step in range(301):
    ix = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([data[i:i + T] for i in ix])          # [B, T]
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])  # 错位一位
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(step, round(loss.item(), 3))
out = model.generate(x[:1, :10], 80)
print("".join(chars[i] for i in out[0].tolist()))
```

这个约 0.84M 参数的模型在 CPU 上几秒就能训练完；损失在 300 步内从约 $\ln V$ 降到大约 0.1–0.2，因为它记住了这段很短的重复文本。在真实语料上，请改为跟踪留出集损失。在真实代码中，请把显式的 softmax 注意力换成 `F.scaled_dot_product_attention(q, k, v, is_causal=True)`，它会调度到融合的 FlashAttention 风格 kernel（[1.6](06-long-context-and-efficient-attention.md)）。

## 实例演算 {#worked-example}

**一个微型注意力计算。** 三个 token，$d_k = 2$。我们计算 token 3 的输出（它可以关注 token 1–3）。设 $q_3 = (1, 0)$，键 $k_1 = (1, 0)$、$k_2 = (0, 1)$、$k_3 = (1, 1)$，值 $v_1 = (1, 0)$、$v_2 = (0, 2)$、$v_3 = (3, 1)$。

- 原始分数 $q_3 \cdot k_j$：$(1, 0, 1)$。除以 $\sqrt{2}$ 缩放后：$(0.707, 0, 0.707)$。
- Softmax：$e^{0.707} = 2.028$，$e^{0} = 1$，所以权重 $= (2.028, 1, 2.028) / 5.056 = (0.401, 0.198, 0.401)$。
- 输出：$0.401\,(1,0) + 0.198\,(0,2) + 0.401\,(3,1) = (1.604, 0.797)$。

token 3 主要混入了 token 1 和 3，它们的键与它的查询对齐。

**7B 级模型的参数量。** 取 Llama 2 7B 的配置：$L = 32$，$d = 4096$，$V = 32{,}000$，SwiGLU FFN 有三个 $d \times d_{ff}$ 的矩阵，$d_{ff} = 11{,}008$（为什么是三个矩阵、为什么是这个怪尺寸，见 [1.4](04-modern-block.md)）。无偏置。

| 组成部分 | 公式 | 数量 |
|---|---|---|
| 每层注意力（$W_Q, W_K, W_V, W_O$） | $4d^2$ | 67,108,864 |
| 每层 FFN | $3 d\, d_{ff}$ | 135,266,304 |
| 全部 32 层 | $32 \times (4d^2 + 3d\,d_{ff})$ | 6,476,005,376 |
| 输入嵌入 + 输出层（不共享） | $2Vd$ | 262,144,000 |
| norm 权重 | $2dL + d$ | 266,240 |
| **总计** | | **6,738,415,616 ≈ 6.74B** |

这与通常所说的"Llama 2 7B"的 6.74B 吻合。对于使用 $4d$ GELU MLP 的 GPT 风格模型，每层参数是 $4d^2 + 8d^2 = 12d^2$，所以 $N \approx 12Ld^2$；GPT-3 175B（$L = 96$，$d = 12{,}288$）给出 $12 \times 96 \times 12{,}288^2 \approx 173.9\text{B}$，外加 0.6B 的嵌入。

**每个 token 的 FLOPs。** 与一个 $m \times n$ 权重的矩阵–向量乘法需要 $mn$ 次乘加 = $2mn$ FLOPs。每个参数在每个 token 上参与一次乘加，所以：

- **前向传播：** 每个 token 约 $2N$ FLOPs。
- **反向传播：** 约 $4N$（对激活和对权重的梯度，各自的开销与前向大致相当）。
- **训练：** 每个 token 约 $6N$ FLOPs。**推理：** 每个生成的 token 约 $2N$。

这忽略了注意力分数本身的计算（$QK^\top$ 和 $AV$），它在上下文长度 $T$ 下，每个 token 增加约 $4 L T d$ 的前向 FLOPs（若利用因果掩码则减半）。对我们的 7B 模型，$T = 4096$ 时是 $4 \times 32 \times 4096 \times 4096 \approx 2.1$ GFLOPs，对比 $2N \approx 13.5$ GFLOPs：已经不容忽视，但在中等上下文长度下，$6N$ 仍是一个不错的首次估算。

**训练成本。** Llama 2 7B 在 2T token 上训练：$6 \times 6.74 \times 10^9 \times 2 \times 10^{12} \approx 8.1 \times 10^{22}$ FLOPs。A100 在 BF16 下的峰值是 312 TFLOP/s；按现实的 40% 利用率算约为 125 TFLOP/s，所以这次训练需要 $8.1 \times 10^{22} / 1.25 \times 10^{14} \approx 6.5 \times 10^8$ GPU 秒 $\approx$ 180,000 GPU 小时。Llama 2 论文报告 7B 模型用了 184,320 个 A100 GPU 小时，所以这个粗略估算在几个百分点之内是准确的。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **仅解码器、Pre-norm** 在前沿大模型中是通用的。编码器–解码器模型（T5）和仅编码器模型（BERT）在嵌入和分类上仍然有用，但不用于前沿的生成。
- **规模。** 从 7B 到 405B 的稠密模型，隐藏宽度通常为 4096–8192，层数 32–126（Llama 3 405B：126 层，$d = 16{,}384$，128 个头）。头大小 $d_h = 128$ 是迄今最常见的选择。
- **偏置**在线性层里大多被去掉了（Llama、DeepSeek）；有些模型在 QKV 投影（Qwen2）或注意力（gpt-oss）中保留偏置。
- **块被升级了，但没有被替换**：用 RMSNorm 代替 LayerNorm，用 RoPE 代替学习式位置，用 SwiGLU 代替 GELU，用 GQA/MLA 代替 MHA，用 MoE 代替稠密 FFN。[1.4](04-modern-block.md) 章逐一实现这些升级。

## 常见误区 {#common-pitfalls-misconceptions}

- **忘了掩码**会让训练损失可疑地飞快降到接近零（模型在抄答案），并得到一个只会生成乱码的模型。
- **reshape 顺序错误。** `x.view(B, T, h, dh).transpose(1, 2)` 是对的；`x.view(B, h, T, dh)` 会悄无声息地把 token 和头搅乱。
- **"注意力就是一切。"** 按参数量计，FFN 通常占主导（标准稠密块里占三分之二）。注意力在位置之间搬运信息；FFN 完成了大部分逐 token 的计算。
- **"6N 是精确的。"** 它忽略了注意力分数的 FLOPs、嵌入查找，以及激活检查点带来的重计算（[1.8](08-training-at-scale.md)）；请把它当作主阶估计。
- **位置信息不是自动就有的。** 没有位置嵌入（或 RoPE）时，因果 Transformer 仍能从掩码中推断出一些顺序，但性能和长度泛化都会变差。

## 本章要点 {#key-takeaways}

- 注意力计算 $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$：每个 token 对更早 token 的值取加权平均，权重取决于查询–键相似度。
- $\sqrt{d_k}$ 因子使分数方差保持为 1，避免 softmax 饱和；因果掩码使下一个 token 的训练可以并行。
- 一个块是 `x + Attn(Norm(x))`，接着是 `x + FFN(Norm(x))`；Pre-norm 加残差连接，才让深层堆叠能稳定训练。
- 参数量 $\approx L(4d^2 + \text{FFN}) + $ 嵌入；训练每个 token 约需 $6N$ FLOPs，推理约 $2N$。
- 这些公式足以把一次 7B 训练估算到与报告的 GPU 小时相差几个百分点之内。

## 自测 {#check-your-understanding}

??? question "为什么没有位置信息的注意力会把输入当作一个集合？"
    分数 $q_i \cdot k_j$ 只取决于 token $i$ 和 $j$ 的内容，而不取决于它们的位置。如果你打乱输入 token 的顺序，同样的分数会以被打乱的顺序出现，输出也以同样方式被打乱。（因果掩码打破了完全的对称性，但只是很弱地打破。）

??? question "当 $d_k = 64$ 且不做缩放时，初始化时注意力分数的标准差是多少（q 和 k 方差为 1）？缩放改变了什么？"
    $\sqrt{64} = 8$。±8 到 ±16 量级的分数会让 softmax 几乎变成 one-hot。除以 $\sqrt{64} = 8$ 把方差恢复为 1。

??? question "估算一个 GPT 风格模型的参数量：$L = 24$，$d = 2048$，$V = 50{,}000$，嵌入共享。"
    $12 L d^2 = 12 \times 24 \times 2048^2 \approx 1.21\text{B}$，再加上共享嵌入的 $Vd \approx 0.10\text{B}$，总共约 1.3B。

??? question "为什么训练每个 token 的开销大约是推理的 3 倍？"
    反向传播需要对激活和对权重的梯度，各自的开销与前向传播大致相当：$2N + 4N = 6N$，而推理是 $2N$。

??? question "Post-LN 在很深的模型中会出什么问题？Pre-LN 改变了什么？"
    在 Post-LN 中，归一化位于残差路径上，初始化时靠近输出层的各层梯度很大，所以训练需要很长的预热，并且可能发散。Pre-LN 让残差路径保持为恒等求和，在任何深度都给出尺度合适的梯度。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 | Transformer：缩放点积注意力、多头注意力、残差连接、位置编码。重点读 §3。 |
| :material-star: [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., 2019 | 仅解码器、pre-norm 的 GPT 配方，本章的代码就遵循它。 |
| :material-book-open-variant: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Xiong et al., 2020 | 解释了为什么 **Pre-LN** 不需要预热技巧也能稳定训练；如今已是通用做法。 |
| :material-book-open-variant: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Touvron et al., 2023 | 实例演算中所用的 7B 配置和 GPU 小时数的来源。 |
