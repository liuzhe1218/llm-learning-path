# 1.6 长上下文与高效注意力

!!! abstract "本章你将学到"
    - 为什么标准注意力在序列长度 $T$ 上的计算量和显存都是 $O(T^2)$，以及这两种开销分别在哪里“咬人”。
    - **FlashAttention** 如何通过分块（tiling）加**在线 softmax**，在不存储 $T \times T$ 矩阵的情况下计算*精确*的注意力（本章给出推导）。
    - 在短序列上训练的模型如何扩展到长序列：**位置插值**、**NTK-aware 缩放**和 **YaRN**。
    - 更便宜的注意力模式：**滑动窗口**、局部/全局交错，以及 **attention sink**。
    - 亚二次复杂度的替代方案：**线性注意力**、**状态空间模型（Mamba）**、**gated delta rule**、**混合**模型和**稀疏注意力**（NSA/DSA）。

## 为什么重要 {#why-this-matters-the-problem}

长上下文让模型能够阅读一整个代码库、一份合同，或者智能体数小时的工具调用记录。但 [1.3](03-transformer.md) 中的注意力要把每个 query 与每个 key 都比较一遍。对于序列长度为 $T$ 的单个头，分数矩阵 $QK^\top$ 有 $T^2$ 个元素。上下文翻倍，这部分工作就变成四倍。当 $T = 131{,}072$（128K）时，一个头的分数矩阵有 $1.7 \times 10^{10}$ 个元素，在 BF16 下占 32 GiB，而这还只是*一层中的一个头*。即使算力够用，朴素地存储这个矩阵也不现实。本章讲两类解决办法：让精确注意力在硬件上高效运行（FlashAttention），或者改变注意力本身，让它不再是二次的（窗口、线性注意力、SSM、稀疏注意力）。本章还会讲，对于位置编码（RoPE，见 [1.4](04-modern-block.md)）只训练到某个长度的模型，如何扩展它的上下文长度。

## 直觉 {#intuition}

- **FlashAttention：** 朴素注意力的问题与其说是算术，不如说是*数据搬运*：把一个巨大的矩阵写进慢速的 GPU 显存，再读回来。FlashAttention 以能放进高速片上存储的小块（tile）来计算注意力，从不把那个大矩阵写出去。让这一切成为可能的窍门，是一种可以增量计算的 softmax：每次处理一块 key。
- **上下文扩展：** RoPE 把位置编码成旋转角度。一个训练到 8K token 的模型从没见过 8K 之后的位置所产生的角度。插值类方法把长位置重新压缩回模型熟悉的角度范围。
- **更便宜的模式：** 大多数 token 大多数时候只需要附近的上下文。让大部分层只看一个局部窗口，再保留少数几层做全注意力，负责远距离检索。
- **线性注意力 / SSM：** 不再保留每一个过去的 key 和 value，而是保留一个固定大小的*摘要*（一个状态矩阵），每个 token 更新一次，就像 RNN。每个 token 的内存变成常数，但摘要是有损的，所以对远处细节的精确回忆会变差。混合模型把两者结合起来。

## 工作原理 {#how-it-works}

### 注意力的开销 {#the-cost-of-attention}

对于一层，有 $h$ 个头、头大小为 $d_h$（因此 $h d_h = d$），序列长度为 $T$：

- **计算：** $QK^\top$ 和 $AV$ 各花 $2 T^2 d$ FLOPs，所以前向传播中每层是 $4 T^2 d$（利用因果掩码可省去约一半）。该层其余部分（投影、FFN）的开销为 $O(T d^2)$。一旦 $T$ 超过 $d$ 的几倍，注意力就占主导。
- **显存（训练）：** 朴素实现会为反向传播存储 $[B, h, T, T]$ 的注意力概率。
- **显存（推理）：** KV cache 随 $T$ 线性增长（见 [1.4](04-modern-block.md)），而且每个解码步都必须读取它的全部内容。

### FlashAttention：分块与在线 softmax {#flashattention-tiling-and-the-online-softmax}

GPU 有容量大但速度慢的主存（HBM，几十 GB，带宽几 TB/s），以及每个流式多处理器（SM）上容量很小但速度很快的片上 SRAM（几百 KB）。朴素注意力把 $S = QK^\top$ 写入 HBM，读回来做 softmax，再写出 $P$，然后为了算 $PV$ 再读一次。FlashAttention（Dao et al., 2022）把 $Q$ 切成行块，把 $K, V$ 切成列块，每次把一个 tile 载入 SRAM，并直接累加输出。障碍在于 softmax 需要*整行*才能归一化：$\mathrm{softmax}(s)_j = e^{s_j} / \sum_{j'} e^{s_{j'}}$。解决办法就是**在线 softmax**。

**推导在线 softmax。** 为了数值稳定，softmax 总是减去最大值后再计算：$\mathrm{softmax}(s)_j = e^{s_j - m} / \ell$，其中 $m = \max_j s_j$，$\ell = \sum_j e^{s_j - m}$。假设我们已经处理了第一块分数，持有它的统计量 $m^{(1)}, \ell^{(1)}$ 以及未归一化的输出 $o^{(1)} = \sum_{j \in \text{chunk 1}} e^{s_j - m^{(1)}} v_j$。第二块到来，其局部最大值为 $\tilde m$。新的全局最大值是 $m^{(2)} = \max(m^{(1)}, \tilde m)$。每个旧的项都是相对于 $m^{(1)}$ 算的；要把它改写成相对于 $m^{(2)}$ 的形式，需要乘以 $e^{m^{(1)} - m^{(2)}}$，因为 $e^{s_j - m^{(1)}} e^{m^{(1)} - m^{(2)}} = e^{s_j - m^{(2)}}$。于是

$$
\ell^{(2)} = e^{m^{(1)} - m^{(2)}}\,\ell^{(1)} + \sum_{j \in \text{chunk 2}} e^{s_j - m^{(2)}}, \qquad
o^{(2)} = e^{m^{(1)} - m^{(2)}}\,o^{(1)} + \sum_{j \in \text{chunk 2}} e^{s_j - m^{(2)}}\, v_j .
$$

处理完最后一块之后，精确的注意力输出就是 $o / \ell$。用大白话说：维护一个滚动的最大值、一个滚动的分母和一个滚动的加权和；每当最大值增大，就把旧的累加器按相应系数缩小。无论 $T$ 多大，每个 query 行只需要 $O(1)$ 个额外的数。

对于反向传播，FlashAttention 每行只存储 $m$ 和 $\ell$（而不存 $T \times T$ 的概率），并在需要时*重新计算* $P$ 的各个 tile。重新计算增加了 FLOPs，但去掉了主要的内存搬运，所以整体更快。结果就是显存为 $O(T)$ 的精确注意力。FlashAttention-2 改进了跨序列和跨 warp 的并行化；FlashAttention-3 针对 Hopper GPU（异步内存拷贝、warp 特化、FP8）。kernel 的细节见 [第 3 部分](../part3/index.md)。

### 扩展上下文窗口 {#extending-the-context-window}

在很长的序列上做预训练代价高昂，所以模型大多在 4K–8K token 上训练，然后在长文档上做一小段继续训练来**扩展**长度。使用 RoPE 时，位置 $m$ 上的第 $i$ 对维度旋转角度为 $m\theta_i$，其中 $\theta_i = b^{-2i/d_h}$（见 [1.4](04-modern-block.md)）。设训练长度为 $L$，目标长度为 $L' = sL$，其中 $s$ 是**缩放因子**。

- **位置插值（PI）**（Chen et al., 2023）：用位置 $m / s$ 代替 $m$，等价于把每个频率都除以 $s$：$\theta_i' = \theta_i / s$。所有角度都保持在训练过的范围内。代价是：用来区分相邻 token 的高频维度对也被压缩了 $s$ 倍，所以相邻位置更难区分。一次短暂的微调（Chen et al. 报告约 1,000 步）就能恢复大部分质量。
- **NTK-aware 缩放：** 改为调整底数，使高频几乎不变，而最低频率恰好被插值 $s$ 倍。最低频率是 $\theta_{d_h/2-1} = b^{-(d_h-2)/d_h}$。要求 $(b')^{-(d_h-2)/d_h} = b^{-(d_h-2)/d_h} / s$，得到

    $$
    b' = b \cdot s^{d_h/(d_h - 2)} .
    $$

    用大白话说：把底数增大到比 $s$ 稍多一点。这样插值是不均匀地分摊的：快的维度对完全不插值，最慢的完全插值。（这个名字来自关于神经正切核（NTK）的论证，涉及如何学习高频。）
- **YaRN**（Peng et al., 2023）把这种划分按频率显式地做出来。对第 $i$ 对维度，计算它的波长 $\lambda_i = 2\pi / \theta_i$ 以及比值 $r_i = L / \lambda_i$（它在训练长度内完成了多少整圈旋转）。转了很多圈的维度对（$r_i > \beta$）在训练中已经见过所有角度：保持不动。转不满一圈的一部分（$r_i < \alpha$）的维度对必须插值：除以 $s$。介于两者之间的，用一段斜坡做线性混合。论文对 Llama 模型取 $\alpha = 1$，$\beta = 32$。YaRN 还会把注意力 logits 乘以一个温度因子 $\sqrt{1/t} = 0.1 \ln s + 1$，以抵消远距离上注意力分布变平的现象。它只需少量继续训练就能达到长上下文，Qwen、DeepSeek-V3、Kimi K2 和 gpt-oss 都在使用。

另一个互补的手段，是从一开始就用较大的底数训练（Llama 3 使用 $b = 500{,}000$），并安排专门的长上下文训练阶段，长度逐步增加。

### 滑动窗口与局部/全局注意力 {#sliding-window-and-localglobal-attention}

**滑动窗口注意力（SWA）** 让每个 token 只关注前面的 $W$ 个 token。每层的计算量降为 $O(T W)$，每层的 KV cache 上限为 $W$ 项。信息传播的距离仍然可以超过 $W$：堆叠 $L$ 层窗口注意力后，感受野最多可达 $L \cdot W$ 个 token，就像堆叠的卷积一样。Mistral 7B 使用 $W = 4096$。

实际中，纯 SWA 会损失精确的远距离检索，所以模型会把局部层和全局层**交错**排列：Gemma 2 以 1:1 交替；Gemma 3 每 1 个全局层配 5 个局部层（窗口 1024）；gpt-oss 则让全注意力与 128 token 的带状（窗口）注意力交替。只有全局层才需要完整长度的 KV cache。

### Attention sink {#attention-sinks}

softmax 的权重之和必须为 1，即使某个头没有任何值得关注的内容。模型学会把这些多余的注意力倾倒在最前面的 token 上，因为每个位置都能看到它们。Xiao et al.（2023）把它们称为 **attention sink**，并指出：如果滑动窗口缓存把最开头的 token 驱逐出去，模型就会崩溃；而保留最前面 4 个 token 加一个最近窗口（StreamingLLM），则能让生成在非常长的流上保持稳定。gpt-oss 把这点内建进了模型：每个头有一个**可学习的 sink logit**，它给 softmax 的分母增加一项，让该头把概率质量放在“什么都不关注”上，而不是放在某个任意的 token 上。

### 线性注意力 {#linear-attention}

把 query $t$ 的因果注意力写成 $o_t = \sum_{j \le t} \mathrm{sim}(q_t, k_j)\, v_j / \sum_{j \le t} \mathrm{sim}(q_t, k_j)$，其中 $\mathrm{sim}(q, k) = e^{q \cdot k / \sqrt{d}}$。**线性注意力**（Katharopoulos et al., 2020）把指数换成特征映射的点积，$\mathrm{sim}(q, k) = \phi(q)^\top \phi(k)$，其中 $\phi$ 为正值函数（如 $\mathrm{elu}(x) + 1$）。于是，利用结合律，

$$
o_t = \frac{\sum_{j \le t} v_j \, \phi(k_j)^\top \phi(q_t)}{\sum_{j \le t} \phi(k_j)^\top \phi(q_t)} = \frac{S_t\, \phi(q_t)}{z_t^\top \phi(q_t)}, \qquad S_t = S_{t-1} + v_t \phi(k_t)^\top, \quad z_t = z_{t-1} + \phi(k_t).
$$

用大白话说：整个历史被压缩进一个 $d_v \times d_k$ 的矩阵 $S_t$（外加一个向量 $z_t$），每个 token 只需往里加一个外积来更新。推理内存与 $T$ 无关（常数），就像 RNN；训练仍然可以并行（或分块）进行。它的弱点是：$S_t$ 是固定大小的和，旧信息永远不会被移除，key 之间会相互干扰。对很久以前的具体事实的回忆因此会退化。

### 状态空间模型与 Mamba {#state-space-models-and-mamba}

一个（离散化的）**状态空间模型**通过隐状态 $h_t$ 把输入序列 $x_t$ 映射为输出：

$$
h_t = \bar A_t\, h_{t-1} + \bar B_t\, x_t, \qquad y_t = C_t\, h_t ,
$$

其中 $\bar A_t$ 控制状态如何衰减，$\bar B_t$ 控制输入如何写入，$C_t$ 控制状态如何被读出。经典 SSM 使用固定（时不变）的矩阵，这使得基于卷积的训练很快，但无法根据内容决定*记住什么*。**Mamba**（Gu & Dao, 2023）让 $\bar B_t$、$C_t$ 和步长（进而包括衰减 $\bar A_t$）成为输入 token 的函数：模型可以有选择地保留或遗忘信息。它用硬件感知的并行扫描（parallel scan）来训练，解码时每个 token 只需 $O(1)$ 内存。**Mamba-2**（Dao & Gu, 2024）证明了这种递推的一个受限形式，等价于一种带衰减掩码的线性注意力（“状态空间对偶”），从而可以做对矩阵乘法友好的分块训练。

### Gated delta rule {#the-gated-delta-rule}

线性注意力是往状态里*加上* $v_t k_t^\top$。可以把 $S$ 看作一个联想记忆：用（单位范数的）key 去读取，得到 $S k$。而 **delta rule** 则是把记忆朝目标值*修正*：它先读出当前的预测 $S_{t-1} k_t$，然后只写入误差，

$$
S_t = S_{t-1} + \beta_t \big(v_t - S_{t-1} k_t\big) k_t^\top = S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,
$$

其中 $\beta_t \in (0, 1]$ 是可学习的、依赖输入的写入强度。当 $\beta_t = 1$ 时，更新之后恰好有 $S_t k_t = v_t$：存放在 key $k_t$ 下的旧值被替换，而不是叠加。**Gated DeltaNet**（Yang et al., 2024）加入了衰减门 $\alpha_t \in (0, 1)$，在上下文发生变化时可以迅速遗忘一切：

$$
S_t = \alpha_t\, S_{t-1}\big(I - \beta_t k_t k_t^\top\big) + \beta_t v_t k_t^\top, \qquad o_t = S_t q_t .
$$

用大白话说：先遗忘一点（门控），再精确覆盖这个 key 所指向的内容（delta），然后写入新的关联。这样既保持了固定大小的状态，又比普通线性注意力有好得多的上下文内回忆能力。

### 混合模型 {#hybrid-models}

线性/SSM 层提供廉价的长上下文；softmax 注意力提供精确的检索。**混合模型**把大部分线性层与少量全注意力层堆叠起来。MiniMax-01 把 lightning（线性）注意力与每隔八层一层的 softmax 注意力交错排列；Qwen3-Next 大约每 3 个 Gated DeltaNet 层配 1 个全注意力层；Jamba 和 NVIDIA 的 Nemotron-H 把 Mamba 层与注意力层混合使用。KV cache 只在那少数几个全注意力层里增长。

### 稀疏注意力 {#sparse-attention}

**稀疏注意力**不去近似注意力，而是在过去 token 的一个*被选中的子集*上计算精确的 softmax 注意力。**原生稀疏注意力（NSA）**（DeepSeek, 2025）为每个 query 结合了三条分支：对压缩后的块摘要做注意力（对全部内容的粗略视图）、对利用这些摘要选出的得分最高的块做注意力（在重要之处看细节），以及滑动窗口（局部上下文），并由可学习的门控混合。它是端到端训练的，并且被设计成让选中的块在内存中连续，从而保持 GPU kernel 的效率。DeepSeek-V3.2 的 **DeepSeek Sparse Attention（DSA）**用一个轻量的“索引器（indexer）”低成本地为过去的 token 打分，然后只对 top-$k$ 做注意力。一个相关的思路是 **Differential Transformer**：保持稠密注意力，但把它算成两个 softmax 图之差，抵消共模的“注意力噪声”，改善长上下文检索。

## 代码 {#code}

**25 行写出 FlashAttention 的前向传播。** 这就是算法本身，用纯 PyTorch 写成（所以并不快）；代码会与朴素注意力做对照检查。

```python
import math, torch

def naive_attention(q, k, v):                     # q, k, v: [T, d]（单头，无掩码）
    s = q @ k.T / math.sqrt(q.shape[-1])          # [T, T] 完整物化
    return torch.softmax(s, dim=-1) @ v

def tiled_attention(q, k, v, block=16):
    """FlashAttention 风格的前向：用在线 softmax 在 K/V 块上流式处理。
    从不物化 [T, T] 矩阵；每个 query 行额外的状态只有 (m, l)。"""
    T, d = q.shape
    scale = 1 / math.sqrt(d)
    out = torch.empty_like(q)
    for i in range(0, T, block):                  # 每个 query 块（对应一个 GPU thread block）
        qi = q[i:i + block] * scale               # [b, d]
        m = torch.full((qi.shape[0], 1), -float("inf"))   # 分数的滚动最大值
        l = torch.zeros(qi.shape[0], 1)                   # exp(score - m) 的滚动和
        acc = torch.zeros(qi.shape[0], d)                 # 滚动的未归一化输出
        for j in range(0, T, block):              # 让 K/V 块流经“SRAM”
            s = qi @ k[j:j + block].T             # [b, b]，只含这个 tile 的分数
            m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
            p = torch.exp(s - m_new)              # tile 的概率（未归一化）
            c = torch.exp(m - m_new)              # 旧统计量的缩放因子
            l = l * c + p.sum(-1, keepdim=True)
            acc = acc * c + p @ v[j:j + block]
            m = m_new
        out[i:i + block] = acc / l                # 最后只归一化一次
    return out

torch.manual_seed(0)
q, k, v = (torch.randn(100, 32) for _ in range(3))
print(torch.allclose(naive_attention(q, k, v), tiled_attention(q, k, v), atol=1e-5))  # True
```

**线性注意力：并行形式 = 递推形式，外加 gated delta rule。**

```python
import torch
import torch.nn.functional as F

def phi(x):                                   # 正值特征映射（Katharopoulos et al.）
    return F.elu(x) + 1

def linear_attention_parallel(q, k, v):       # 每个都是 [T, d]；因果，训练风格
    A = phi(q) @ phi(k).T                     # [T, T] 相似度（不做 softmax）
    A = torch.tril(A)                         # 因果掩码
    return (A @ v) / A.sum(-1, keepdim=True)

def linear_attention_recurrent(q, k, v):      # 推理风格：每步 O(1) 状态
    d_k, d_v = q.shape[1], v.shape[1]
    S = torch.zeros(d_v, d_k)                 # 状态：sum_j v_j phi(k_j)^T
    z = torch.zeros(d_k)                      # 归一化项：sum_j phi(k_j)
    outs = []
    for t in range(q.shape[0]):
        S = S + torch.outer(v[t], phi(k[t]))
        z = z + phi(k[t])
        outs.append(S @ phi(q[t]) / (z @ phi(q[t])))
    return torch.stack(outs)

def gated_delta_recurrent(q, k, v, alpha, beta):
    """S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T ; o_t = S_t q_t"""
    S = torch.zeros(v.shape[1], k.shape[1])
    outs = []
    for t in range(q.shape[0]):
        kt = F.normalize(k[t], dim=0)         # 单位范数的 key 让更新保持稳定
        S = alpha[t] * (S - beta[t] * torch.outer(S @ kt, kt)) + beta[t] * torch.outer(v[t], kt)
        outs.append(S @ q[t])
    return torch.stack(outs)

torch.manual_seed(0)
T, d = 50, 16
q, k, v = (torch.randn(T, d) for _ in range(3))
print(torch.allclose(linear_attention_parallel(q, k, v),
                     linear_attention_recurrent(q, k, v), atol=1e-5))        # True
o = gated_delta_recurrent(q, k, v, alpha=torch.full((T,), 0.95), beta=torch.full((T,), 0.5))

# 把 delta rule 当作联想记忆：把同一个 (key -> value) 写两次不会累加。
S = torch.zeros(d, d)
key, val = F.normalize(torch.randn(d), dim=0), torch.randn(d)
for _ in range(3):
    S = S - torch.outer(S @ key, key) + torch.outer(val, key)   # beta = 1, alpha = 1
print(torch.allclose(S @ key, val, atol=1e-5))                  # True：精确回忆
```

如果用普通线性注意力把同一对写三次，读出来会是该值的 $3 \times$；delta rule 读出来则恰好等于原值。

## 实例演算 {#worked-example}

**手算在线 softmax。** 某个 query 的分数为 $s = (1, 2 \mid 4)$，切成长度 2 和 1 的两块，标量值为 $v = (10, 20 \mid 30)$。

- 第 1 块：$m^{(1)} = 2$，$\ell^{(1)} = e^{1-2} + e^{0} = 1.3679$，$o^{(1)} = e^{-1} \cdot 10 + 1 \cdot 20 = 23.679$。
- 第 2 块：局部最大值为 4，所以 $m^{(2)} = 4$，缩放因子为 $e^{2-4} = 0.1353$。
- $\ell^{(2)} = 1.3679 \times 0.1353 + e^{0} = 1.1851$；$o^{(2)} = 23.679 \times 0.1353 + 30 = 33.205$。
- 输出 $= 33.205 / 1.1851 = 28.02$。

直接验算：$\big(e^{-3} \cdot 10 + e^{-2} \cdot 20 + 30\big) / \big(e^{-3} + e^{-2} + 1\big) = 33.205 / 1.1851 = 28.02$。完全一致。

**局部/全局交错的显存。** 考虑一个假设的模型：62 层，16 个 KV 头，头大小 128，BF16 缓存，128K 上下文。每层中，每个被缓存的 token 占 $2 \times 16 \times 128 \times 2 = 8192$ 字节。

- 62 层全部为全局：每个序列 $62 \times 8192 \times 131{,}072 = 62$ GiB。
- 10 个全局层 + 52 个窗口为 1024 token 的局部层：$10 \times 8192 \times 131{,}072 + 52 \times 8192 \times 1024 \approx 10.0 + 0.4 = 10.4$ GiB。

约 6× 的缩减，这就是 Gemma 3 从 Gemma 2 的 1:1 模式改为 5:1 并采用较短窗口的原因。

**上下文扩展的数字。** 从 32K 扩展到 128K，意味着 $s = 4$。取 $d_h = 128$、$b = 10{,}000$，NTK-aware 缩放给出 $b' = 10{,}000 \times 4^{128/126} \approx 40{,}890$。YaRN 的注意力温度因子为 $0.1 \ln 4 + 1 \approx 1.139$。

## 实践中的做法（2025–2026） {#in-practice-20252026}

- **FlashAttention 风格的 kernel 已经无处不在**，训练和 prefill 都用它；没有人再去物化注意力矩阵。在 PyTorch 中，`F.scaled_dot_product_attention` 会自动选择这类 kernel。
- **128K 的上下文长度已是开源前沿模型的标配**（Llama 3、DeepSeek-V3、配合 YaRN 的 Qwen3、gpt-oss、Kimi K2、Gemma 3），通常的做法是先短上下文预训练，再分阶段扩展到长上下文。Llama 3 分六个增量阶段完成了到 128K 的长上下文训练。
- **减小 KV cache 是主要的架构抓手**：GQA/MLA（见 [1.4](04-modern-block.md)）加上局部/全局交错（Gemma 3、gpt-oss）。
- **混合模型正在进入前沿模型**（MiniMax-01、Qwen3-Next、Nemotron-H），而可训练的稀疏注意力（DeepSeek-V3.2 中的 NSA、DSA）则是在长距离上保持精确 softmax 注意力的主要替代方案。
- **上下文并行**（把一个序列切分到多块 GPU 上，例如 Ring Attention）是百万 token 训练能放进显存的办法；见 [1.8](08-training-at-scale.md)。

## 常见误区 {#common-pitfalls-misconceptions}

- **“FlashAttention 是一种近似。”** 它是精确的（至多存在浮点运算顺序不同带来的差别）；它改变的是内存访问模式，而不是数学。
- **“FlashAttention 让注意力变成线性的。”** 显存变成了 $O(T)$，但计算量仍然是 $O(T^2)$。
- **声称的上下文 ≠ 可用的上下文。** 模型可能接受 128K token，却仍然无法利用中间部分的信息。请在你需要的长度上，用检索和推理任务来评测。
- **在推理时更改 RoPE 缩放，却与训练不匹配。** 插值方法在配合它们设计时所用的继续训练时效果最好（甚至只有那时才有效）；设置必须与检查点的配置一致。
- **期望线性注意力像 softmax 注意力那样回忆。** 固定大小的状态无法存储无限多的事实；这正是混合模型保留部分全注意力的原因。
- **把最开头的 token 从滑动缓存中驱逐**，会让依赖 attention sink 的模型崩溃。

## 本章要点 {#key-takeaways}

- 注意力的计算量是 $O(T^2)$；朴素实现的显存同样是 $O(T^2)$，FlashAttention 通过分块和在线 softmax 消除了后者。
- 在线 softmax 维护滚动的最大值、分母和输出，每当最大值增大，就按 $e^{m_{\text{old}} - m_{\text{new}}}$ 缩放。
- 上下文扩展就是重新缩放 RoPE 的频率：均匀缩放（PI）、通过底数（NTK）、或按频段分别处理并加上温度（YaRN）。
- 局部窗口与少量全局层交错，可以大幅削减 KV cache；attention sink 让窗口缓存保持稳定。
- 线性注意力、SSM 和 gated delta rule 用固定大小的状态取代不断增长的 KV cache；混合模型和可训练的稀疏注意力则把效率与精确检索结合起来。

## 自测 {#check-your-understanding}

??? question "FlashAttention 在反向传播中多做了 FLOPs，为什么仍然能加速？"
    注意力受限于内存搬运，而不是算术。在高速 SRAM 里重新计算 tile，比把 $T \times T$ 矩阵写入 HBM 再读出来更便宜。

??? question "在线 softmax 中，如果新一块的最大值比滚动最大值小，会发生什么？"
    $m$ 保持不变，于是缩放因子 $e^{m_{\text{old}} - m_{\text{new}}} = 1$，旧的累加器不变；新的项直接加进去即可。

??? question "一个模型有 32 层，每层的滑动窗口为 4096。信息最多能向最后一个 token 传播多远？"
    最多约 $32 \times 4096 = 131{,}072$ 个 token，因为每一层最多能把信息移动一个窗口的距离。实际上，经过多次跳转后信息会退化，这就是为什么要保留全局层。

??? question "为什么 PI 在短距离上比 NTK/YaRN 差？"
    PI 把所有频率都除以 $s$，包括用来区分相邻 token 的高频，所以局部位置分辨率下降了 $s$ 倍。NTK/YaRN 几乎不动高频，只对低频做插值。

??? question "当同一个 key 被写入两次、对应不同的值时，delta rule 与线性注意力有什么不同？"
    线性注意力把两个值都加进去，所以读出来是它们的和。delta rule（取 $\beta = 1$）会用第二个值替换第一个，因为它只写入目标值与当前读出值之间的差。

## 延伸阅读 {#further-reading}

| 论文 | 为什么读 |
|---|---|
| :material-star: [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., 2022 | 对注意力分块，使 $T \times T$ 矩阵从不触及 HBM；精确、更快、显存 $O(T)$。 |
| :material-book-open-variant: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — Milakov & Gimelshein, 2018 | FlashAttention 的在线 softmax 所依据的单遍 softmax 归一化方法。 |
| :material-book-open-variant: [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Dao, 2023 | 在 warp/thread block 之间更好的工作划分；比 v1 快约 2×。 |
| :material-book-open-variant: [FlashAttention-3](https://arxiv.org/abs/2407.08608) — Shah et al., 2024 | 针对 Hopper：异步（TMA、warp 特化）与 FP8。 |
| :material-book-open-variant: [Extending Context Window of LLMs via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Chen et al., 2023 | RoPE 的位置插值，以及它为什么优于外推。 |
| :material-star: [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) — Peng et al., 2023 | 按频率（“NTK-by-parts”）重新缩放 RoPE，加上注意力温度；应用广泛。 |
| :material-book-open-variant: [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889) — Liu et al., 2023 | 把序列切分到多个设备上，并让 K/V 块沿环传递：上下文并行的基础。 |
| :material-book-open-variant: [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023 | 在一个强大的小模型里使用带滚动 KV cache 的滑动窗口注意力。 |
| :material-book-open-variant: [Gemma 2](https://arxiv.org/abs/2408.00118) — Gemma Team, 2024 | 局部/全局交错注意力与 logit 软截断（soft-capping）。 |
| :material-star: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — Xiao et al., 2023 | 发现了 attention sink；StreamingLLM 保留起始 token 加一个窗口。 |
| :material-book-open-variant: [Transformers are RNNs (Linear Attention)](https://arxiv.org/abs/2006.16236) — Katharopoulos et al., 2020 | 核化注意力及其递推形式，如上文所推导。 |
| :material-star: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023 | 依赖输入的 SSM；训练 $O(T)$，推理时每个 token 的内存为 $O(1)$。 |
| :material-book-open-variant: [Transformers are SSMs (Mamba-2)](https://arxiv.org/abs/2405.21060) — Dao & Gu, 2024 | SSM 与线性注意力之间的状态空间对偶；更快的基于矩阵乘法的算法。 |
| :material-new-box: [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — Yang et al., 2024 | 线性注意力的门控加 delta rule；以混合形式用于 Qwen3-Next。 |
| :material-new-box: [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) — MiniMax, 2025 | 第一个前沿规模的混合线性注意力 MoE（456B），上下文最长达 1M–4M。 |
| :material-new-box: [Native Sparse Attention (NSA)](https://arxiv.org/abs/2502.11089) — Yuan et al. (DeepSeek), 2025 | 与硬件对齐、可训练的稀疏注意力；V3.2 中 DeepSeek Sparse Attention 的前身。 |
| :material-book-open-variant: [Differential Transformer](https://arxiv.org/abs/2410.05258) — Ye et al., 2024 | 把注意力写成两个 softmax 图之差，抵消注意力噪声。 |
